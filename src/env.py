import gym
import rospy
import numpy as np
import gurobipy as gp
from gym.utils import seeding
from geometry_msgs.msg import Twist
from px4_mavros import Px4Controller
from gazebo_msgs.srv import SetModelState
from gazebo_msgs.msg import ModelState, ModelStates
from pyquaternion import Quaternion

class Env(gym.Env):
    def __init__(self):
        # QP problem
        self.m = gp.Model("qp")
        self.m.setParam("LogToConsole",0)
        self.x = self.m.addVars(3,ub=0.3, lb=-0.3, name="x")
        self.height_l = 0.3
        self.height_u = 1.5
        self.d_safe_car = 0.7
        self.gamma = 1.0

        # Gym-Like variable
        self._min_velocity, self._max_velocity = -0.3, 0.3
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(3+2,), dtype=np.float32)
        self.action_space = gym.spaces.Box(self._min_velocity, self._max_velocity, shape=(3,), dtype=np.float32)

        self.state_sub = rospy.Subscriber('/gazebo/model_states', ModelStates, self.odom, queue_size=10)
        self.set_model_proxy = rospy.ServiceProxy("/gazebo/set_model_state", SetModelState)

        self.P1, self.Pb = None, None
        self.Pb_origin = np.array([-3, 0, 0.5])
        self.px4_bearing = Px4Controller("iris_bearing")
        self.control_rate = rospy.Rate(30)

        self.seed()
        self._initialize()

    def reset(self):
        self._move_to("car1", [0, 0, 0])

        rospy.set_param("/iris_bearing/vel_control", 0)
        while np.linalg.norm(self.Pb-self.Pb_origin) > 0.1:
            self.control_rate.sleep()

        rospy.set_param("/car_navigation/start", 1)

        return np.concatenate([self.Pb, [0, 0]])  # [pb_x, pb_y, pb_z, p1_x, p1_y]

    def step(self, action):
        action = self._qpsolver(action)
        bearing_cmd_vel = Twist()
        bearing_cmd_vel.linear.x = action[0]
        bearing_cmd_vel.linear.y = action[1]
        bearing_cmd_vel.linear.z = action[2]

        self.px4_bearing.vel_control(bearing_cmd_vel)
        self.control_rate.sleep()

        reward = self._get_reward()

        done = False
        if rospy.get_param("/car_navigation/start") == 0:
            done = True

        info = None

        next_state = np.concatenate([self.Pb, self.P1[:2]])

        return next_state, reward, done, info

    def close(self):
        rospy.set_param("iris_bearing/vel_control", 0)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def odom(self, msg):
        UAV1_index = msg.name.index('iris_bearing')
        car1_index = msg.name.index('car1')

        self.Pb = np.array([msg.pose[UAV1_index].position.x, msg.pose[UAV1_index].position.y, msg.pose[UAV1_index].position.z])
        self.P1 = np.array([msg.pose[car1_index].position.x, msg.pose[car1_index].position.y, msg.pose[car1_index].position.z])

        self.A = np.array([ \
                           (-2*(self.Pb-self.P1)[:2]).tolist()+[0], \
                           [0]*2+[-1], \
                           [0]*2+[1] \
                           ])

        self.b = np.array([ \
                           np.linalg.norm((self.Pb-self.P1)[:2])**2 - self.d_safe_car**2, \
                           self.Pb[2] - self.height_l, \
                           self.height_u - self.Pb[2] \
                           ])*self.gamma

    def _get_reward(self):
        """
        TODO
        """
        min_r_xy = 0.7
        min_r = np.sqrt(0.3**2 + 0.7**2)
        max_obs = 1 / (min_r_xy * min_r)

        P1, Pb = self.P1, self.Pb
        r_xy = np.sqrt( (P1[0] - Pb[0])**2 + (P1[1] - Pb[1])**2 )
        r = np.sqrt( (P1[0] - Pb[0])**2 + (P1[1] - Pb[1])**2 + (P1[2] - Pb[2])**2 )
        obs = 1 / (r_xy * r)

        reward = obs / max_obs

        return reward

    def _move_to(self, name, pos):
        q = Quaternion(axis=[0, 0, 1], angle=np.random.uniform(-np.pi, np.pi))
        state_msg = ModelState()
        state_msg.model_name = name
        state_msg.pose.position.x = pos[0]
        state_msg.pose.position.y = pos[1]
        state_msg.pose.position.z = pos[2]
        state_msg.pose.orientation.x = q.x
        state_msg.pose.orientation.y = q.y
        state_msg.pose.orientation.z = q.z
        state_msg.pose.orientation.w = q.w
        rospy.wait_for_service("/gazebo/set_model_state")
        try:
            self.set_model_proxy(state_msg)
        except (rospy.ServiceException) as e:
            exit("\033[91m" + "[ERROR] /gazebo/set_model_state service call failed" + "\033[0m")

    def _initialize(self):
        while self.P1 is None or self.Pb is None:
            self.control_rate.sleep()

    def _qpsolver(self, cmd):
        A, b = self.A, self.b

        obj = ((self.x[0] - cmd[0])**2 + (self.x[1] - cmd[1])**2 + (self.x[2] - cmd[2])**2) # optimal
        self.m.setObjective(obj, gp.GRB.MINIMIZE)

        self.m.remove(self.m.getConstrs())

        for i in range (b.size):
            self.m.addConstr(A[i,0]*self.x[0] + A[i,1]*self.x[1] + A[i,2]*self.x[2] <= b[i], "c"+str(i))

        self.m.optimize()
        if self.m.status == gp.GRB.OPTIMAL:
            optimal = self.m.getVars()
            action = np.array([optimal[0].x, optimal[1].x, optimal[2].x])
        else:
            action = np.array([0, 0, 0])

        return action

if __name__ == "__main__":
    rospy.init_node("env_node")

    EPISODE_NUM = 3
    MAX_TIMESTEPS = int(1e7)

    env = Env()
    for ep in range(EPISODE_NUM):
        print(f"Episode: {ep}")
        state = env.reset()
        for t in range(MAX_TIMESTEPS):
            action = env.action_space.sample()
            next_state, reward, done, _ = env.step(action)
            if done:
                break
            state = next_state
    env.close()
