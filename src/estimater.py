#!/usr/bin/env python

import rospy
from ukf import UKF
import numpy as np
from math import sin,cos,sqrt,atan2
from gazebo_msgs.msg import ModelStates
from std_msgs.msg import Float64MultiArray
from rosgraph_msgs.msg import Clock
import rosbag
bag = rosbag.Bag('rl_based.bag', 'w')

msgs = None
P1,Pb,measurement = None,None,None
time_last = 0
det_covariance,all_state = Float64MultiArray(),Float64MultiArray()
clock = Clock()

# Process Noise
q = np.eye(6)
q[0][0] = 0.0001 # target-x 
q[1][1] = 0.0001 # target-y
q[2][2] = 0.0001 # target-z
q[3][3] = 0.0001 # target-vx
q[4][4] = 0.0001 # target-vy
q[5][5] = 0.0001 # target-vz


# create measurement noise covariance matrices
r_measurement = np.eye(3)
r_measurement[0][0] = 0.0005
r_measurement[1][1] = 0.0005
r_measurement[2][2] = 0.0005

# create initial matrices
ini = np.array([1,1,1,0,0,0])

def iterate_x(x_in, timestep, inputs):
    '''this function is based on the x_dot and can be nonlinear as needed'''
    ret = np.zeros(len(x_in))
    ret[0] = x_in[0] + timestep * x_in[3]
    ret[1] = x_in[1] + timestep * x_in[4]
    ret[2] = x_in[2] + timestep * x_in[5]
    ret[3] = x_in[3]
    ret[4] = x_in[4]
    ret[5] = x_in[5]

    return ret

def measurement_model(x_in, data):
    """
    :param x_in: states
    :param data: UAV positions of c,r,b and thetac
    """
    ret = np.zeros(3)
    ret[0] = sqrt((x_in[0] - data[0])**2 + (x_in[1] - data[1])**2 + (x_in[2] - data[2])**2)
    ret[1] = atan2(x_in[1] - data[1],x_in[0] - data[0])
    ret[2] = atan2(x_in[2] - data[2],sqrt((x_in[0] - data[0])**2 + (x_in[1] - data[1])**2))
    return ret

def odom(msg):
    global P1,Pb,msgs
    msgs = msg
    UAV1_index = msg.name.index('iris_bearing')
    car1_index = msg.name.index('car1')

    Pb = np.array([msg.pose[UAV1_index].position.x, msg.pose[UAV1_index].position.y, msg.pose[UAV1_index].position.z])
    P1 = np.array([msg.pose[car1_index].position.x, msg.pose[car1_index].position.y, msg.pose[car1_index].position.z])

def add_measurementnoise():
    global measurement

    measurement = measurement_model(state, uav_state)
    measurement[[0,1,2]] += np.random.normal(0,0.02,3)

def ukf():
    global time_last

    d_t = rospy.Time.now().to_sec() - time_last
    state_estimator.predict(d_t)
    state_estimator.update(3, measurement, r_measurement, uav_state)
    time_last = rospy.Time.now().to_sec()

def clock_cb(msg):
    global clock
    clock = msg

if __name__ == "__main__":
    try:
        rospy.init_node('estimate')
        clock_sub = rospy.Subscriber("/clock", Clock, clock_cb, queue_size=10)
        state_pub = rospy.Publisher("/state", Float64MultiArray, queue_size=10)

        # pass all the parameters into the UKF!
        # number of state variables, process noise, initial state, initial coariance, three tuning paramters, and the iterate function
        state_estimator = UKF(6, q, ini, 0.01*np.eye(6), 0.001, 0.0, 2.0, iterate_x,measurement_model)
        rospy.Subscriber('/gazebo/model_states', ModelStates, odom, queue_size=10)
        rate = rospy.Rate(400)
        while P1 is None:
            rate.sleep()
        while rospy.get_param("/car_navigation/start") != 1:
            rate.sleep()

        while not rospy.is_shutdown():
            if rospy.get_param("/car_navigation/start") == 0:
                break
            '''
            msgs = rospy.wait_for_message('/gazebo/model_states', ModelStates)
            odom(msgs)
            '''
            state = np.array([P1[0],P1[1],P1[2],0,0,0])
            uav_state = Pb
            add_measurementnoise()
            ukf()
            estimate_state = state_estimator.get_state()
            all_state.data = list(state)+list(uav_state)
            state_pub.publish(all_state)
#            print "Estimated state: ", state_estimator.get_state()
#            print "Covariance: ", np.linalg.det(state_estimator.get_covar())
            position_covar = state_estimator.get_covar()
            position_covar = np.delete(position_covar,[3,4,5],axis=1)
            position_covar = np.delete(position_covar,[3,4,5],axis=0)
            det_covariance.data = [np.linalg.det(position_covar),np.linalg.norm(P1-estimate_state[:3])]
            print(det_covariance.data)
            print('--')
            bag.write('det_covariance', det_covariance)
            bag.write('/gazebo/model_states', msgs)
            bag.write('/clock', clock)
#            break
            rate.sleep()
    except rospy.ROSInterruptException:
        pass
    finally:
        bag.close()
