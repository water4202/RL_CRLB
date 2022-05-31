import gym
import rospy
import numpy as np
from geometry_msgs.msg import Twist
from px4_mavros import Px4Controller
from gazebo_msgs.srv import SetModelState
from gazebo_msgs.msg import ModelState, ModelStates

class Env(gym.Env):
    def __init__(self):
        self.state_sub = rospy.Subscriber('/gazebo/model_states', ModelStates, self.odom, queue_size=10)
        self.set_model_proxy = rospy.ServiceProxy("/gazebo/set_model_state", SetModelState)

        self.P1, self.Pb = None, None
        self.Pb_origin = np.array([-3, 0, 0.5])
        self.px4_bearing = Px4Controller("iris_bearing")
        self.control_rate = rospy.Rate(30)

        # Gym-Like variable
        self._min_velocity, self._max_velocity = -1.3, 1.3
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(3+2,), dtype=np.float32)
        self.action_space = gym.spaces.Box(self._min_velocity, self._max_velocity, shape=(4,), dtype=np.float32)

        self._initialize()

    def reset(self):
        self._move_to("car1", [0, 0, 0])

        rospy.set_param("/iris_bearing/vel_control", 0)
        while np.linalg.norm(self.Pb-self.Pb_origin) > 0.1:
            self.control_rate.sleep()

        rospy.set_param("/car_navigation/start", 1)

        return np.concatenate([self.Pb, [0, 0]])  # [pb_x, pb_y, pb_z, p1_x, p1_y]

    def step(self, action):
        bearing_cmd_vel = Twist()
        bearing_cmd_vel.linear.x = action[0]
        bearing_cmd_vel.linear.y = action[1]
        bearing_cmd_vel.linear.z = action[2]
        bearing_cmd_vel.angular.z = action[3]

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

    def odom(self, msg):
        UAV1_index = msg.name.index('iris_bearing')
        car1_index = msg.name.index('car1')

        self.Pb = np.array([msg.pose[UAV1_index].position.x, msg.pose[UAV1_index].position.y, msg.pose[UAV1_index].position.z])
        self.P1 = np.array([msg.pose[car1_index].position.x, msg.pose[car1_index].position.y, msg.pose[car1_index].position.z])

    def _get_reward(self):
        """
        TODO
        """
        return 0

    def _move_to(self, name, pos):
        state_msg = ModelState()
        state_msg.model_name = name
        state_msg.pose.position.x = pos[0]
        state_msg.pose.position.y = pos[1]
        state_msg.pose.position.z = pos[2]
        rospy.wait_for_service("/gazebo/set_model_state")
        try:
            self.set_model_proxy(state_msg)
        except (rospy.ServiceException) as e:
            exit("\033[91m" + "[ERROR] /gazebo/set_model_state service call failed" + "\033[0m")

    def _initialize(self):
        while self.P1 is None or self.Pb is None:
            self.control_rate.sleep()

if __name__ == "__main__":
    rospy.init_node("env_node")

    EPISODE_NUM = 3
    MAX_TIMESTEPS = int(1e7)

    env = Env()
    for ep in range(EPISODE_NUM):
        print(ep)
        state = env.reset()
        for t in range(MAX_TIMESTEPS):
            action = env.action_space.sample()
            next_state, reward, done, _ = env.step(action)
            if done:
                break
            state = next_state
            # print(reward)
    env.close()
