#!/usr/bin/env python

import rospy
from ukf import UKF
import numpy as np
from math import sin,cos,sqrt,atan2
from gazebo_msgs.msg import ModelStates
from std_msgs.msg import Float64MultiArray
from rosgraph_msgs.msg import Clock
import rosbag
bag = rosbag.Bag('crlb_based.bag', 'w')

msgs = None
P1,Pc,measurement,thetac = None,None,None,None
fx,fy,Cu,Cv = 565.6,565.6,320.0,240.0
time_last = 0
det_covariance,all_state = Float64MultiArray(),Float64MultiArray()
clock = Clock()

# Process Noise
q = np.eye(4)
q[0][0] = 0.0001 # target1-x 
q[1][1] = 0.0001 # target1-y
q[2][2] = 0.0001 # target1-vx
q[3][3] = 0.0001 # target1-vy

# create measurement noise covariance matrices
r_measurement = np.eye(2)
r_measurement[0][0] = 4
r_measurement[1][1] = 4

# create initial matrices
ini = np.array([1,1,0,0])

def iterate_x(x_in, timestep, inputs):
    '''this function is based on the x_dot and can be nonlinear as needed'''
    ret = np.zeros(len(x_in))
    ret[0] = x_in[0] + timestep * x_in[2]
    ret[1] = x_in[1] + timestep * x_in[3]
    ret[2] = x_in[2]
    ret[3] = x_in[3]

    return ret

def measurement_model(x_in, data):
    """
    :param x_in: states
    :param data: UAV positions of c,r,b and thetac
    """
    ret = np.zeros(2)
    ret[0] = fx*(sin(data[3])*(x_in[0] - data[0]) - cos(data[3])*(x_in[1] - data[1]))/(cos(data[3])*(x_in[0] - data[0]) + sin(data[3])*(x_in[1] - data[1])) + Cu
    ret[1] = -fy*(x_in[2] - data[2])/(cos(data[3])*(x_in[0] - data[0]) + sin(data[3])*(x_in[1] - data[1])) + Cv
    return ret

def odom(msg):
    global P1,Pc,msgs
    msgs = msg
    UAV1_index = msg.name.index('iris_camera')
    car1_index = msg.name.index('car1')

    Pc = np.array([msg.pose[UAV1_index].position.x, msg.pose[UAV1_index].position.y, msg.pose[UAV1_index].position.z])
    P1 = np.array([msg.pose[car1_index].position.x, msg.pose[car1_index].position.y, msg.pose[car1_index].position.z])

def add_measurementnoise():
    global measurement

    measurement = measurement_model(state, uav_state)
    measurement[[0,1]] += np.random.normal(0,2,2)

def ukf():
    global time_last

    d_t = rospy.Time.now().to_sec() - time_last
    state_estimator.predict(d_t)
    state_estimator.update(2, measurement, r_measurement, uav_state)
    time_last = rospy.Time.now().to_sec()

def theta_update(msg):
    global thetac
    thetac = msg.data[0]

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
        state_estimator = UKF(4, q, ini, 0.01*np.eye(4), 0.001, 0.0, 2.0, iterate_x,measurement_model)
        rospy.Subscriber('/gazebo/model_states', ModelStates, odom, queue_size=10)
        thetac_sub = rospy.Subscriber("/theta_iris_camera", Float64MultiArray, theta_update, queue_size=10)
        while thetac == None:
            pass
        rate = rospy.Rate(400)
        while not rospy.is_shutdown():
            '''
            msgs = rospy.wait_for_message('/gazebo/model_states', ModelStates)
            odom(msgs)
            '''
            state = np.array([P1[0],P1[1],0,0])
            uav_state = np.array([Pc[0],Pc[1],Pc[2],thetac])
            add_measurementnoise()
            ukf()
            estimate_state = state_estimator.get_state()
            all_state.data = list(estimate_state)+list(uav_state)
            state_pub.publish(all_state)
#            print "Estimated state: ", state_estimator.get_state()
#            print "Covariance: ", np.linalg.det(state_estimator.get_covar())
            position_covar = state_estimator.get_covar()
            position_covar = np.delete(position_covar,[2,3],axis=1)
            position_covar = np.delete(position_covar,[2,3],axis=0)
            det_covariance.data = [np.linalg.det(position_covar),np.linalg.norm([P1[0],P1[1]]-estimate_state[:2])]
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
