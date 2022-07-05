#!/usr/bin/python

import rosbag
import matplotlib.pyplot as plt
import numpy as np
from math import sqrt

sigma = 0.02
optimal = [sigma**2*0.7*sqrt(0.3**2 + 0.7**2)]*50

bag = rosbag.Bag('crlb_based.bag')
crlb_covariance=[]
crlb_p1_error=[]
crlb_time = []
crlb = []

for topic, msg, t in bag.read_messages(topics='det_covariance'):

	crlb_covariance.append(msg.data[0])
	crlb_p1_error.append(msg.data[1])
	crlb_time.append(float(t.secs) + float(t.nsecs)/10**9)

for topic, msg, t in bag.read_messages(topics='/gazebo/model_states'):

	UAV1_index = msg.name.index('iris_bearing')
	car1_index = msg.name.index('car1')

	Pb = np.array([msg.pose[UAV1_index].position.x, msg.pose[UAV1_index].position.y, msg.pose[UAV1_index].position.z])
	P1 = np.array([msg.pose[car1_index].position.x, msg.pose[car1_index].position.y, msg.pose[car1_index].position.z])

	crlb.append(sigma**2*np.linalg.norm((Pb-P1)[:2])*np.linalg.norm(Pb-P1))
	
bag.close()
ini = crlb_time[0]
crlb_time[:] = [x - ini for x in crlb_time]

bag = rosbag.Bag('rl_based.bag')
rl_covariance=[]
rl_p1_error=[]
rl_time = []
rl = []

for topic, msg, t in bag.read_messages(topics='det_covariance'):

	rl_covariance.append(msg.data[0])
	rl_p1_error.append(msg.data[1])
	rl_time.append(float(t.secs) + float(t.nsecs)/10**9)

for topic, msg, t in bag.read_messages(topics='/gazebo/model_states'):

	UAV1_index = msg.name.index('iris_bearing')
	car1_index = msg.name.index('car1')

	Pb = np.array([msg.pose[UAV1_index].position.x, msg.pose[UAV1_index].position.y, msg.pose[UAV1_index].position.z])
	P1 = np.array([msg.pose[car1_index].position.x, msg.pose[car1_index].position.y, msg.pose[car1_index].position.z])

	rl.append(sigma**2*np.linalg.norm((Pb-P1)[:2])*np.linalg.norm(Pb-P1))

bag.close()
ini = rl_time[0]
rl_time[:] = [x - ini for x in rl_time]

crlb_p1_avg_error = sum(crlb_p1_error[20:12432])/(len(crlb_p1_error)-20-1706)
print(crlb_p1_avg_error)
crlb_p1_avg_error = [crlb_p1_avg_error]*len(crlb_p1_error)

rl_p1_avg_error = sum(rl_p1_error[20:])/(len(rl_p1_error)-20)
print(rl_p1_avg_error)
rl_p1_avg_error = [rl_p1_avg_error]*len(rl_p1_error)

crlb_avg_covariance = sum(crlb_covariance[20:12432])/(len(crlb_covariance)-20-1706)
print(crlb_avg_covariance)

rl_avg_covariance = sum(rl_covariance[20:])/(len(rl_covariance)-20)
print(rl_avg_covariance)

plt.plot(crlb_time,crlb,"r",label="crlb_based")
plt.plot(rl_time,rl,"b",label="rl_based")
plt.plot(optimal, "k", linestyle='dashed')
plt.ylabel('CRLB')
plt.xlabel('time')
plt.ylim(0, 0.004)
plt.xlim(-0.5, 30)
plt.legend()
plt.show()

plt.plot(crlb_time,crlb_covariance,"r",label="crlb_based")
plt.plot(rl_time,rl_covariance,"b",label="rl_based")
plt.ylabel('det(P)')
plt.xlabel('time')
plt.ylim(1e-16, 1e-13)
plt.xlim(-0.5, 30)
plt.legend()
plt.show()
'''
plt.plot(optimal_time,optimal_covariance,"r")
plt.plot(worst_time,worst_covariance,"b")
plt.ylim(1e-48, 1e-44)
plt.xlim(0, 58)
plt.show()
'''

plt.plot(crlb_time,crlb_p1_error,"r",label="crlb_based")
plt.plot(crlb_time,crlb_p1_avg_error,"k",linestyle="dashed",label="crlb_based_mean")
plt.plot(rl_time,rl_p1_error,"b",label="rl_based")
plt.plot(rl_time,rl_p1_avg_error,"k",linestyle="dotted",label="rl_based_mean")
plt.ylabel('p1_error')
plt.xlabel('time')
plt.ylim(0, 0.3)
plt.xlim(-0.5, 30)
plt.legend()
plt.show()
'''
plt.plot(optimal_time,optimal_p1_error,"r")
plt.plot(worst_time,worst_p1_error,"b")
plt.ylim(0, 0.1)
plt.xlim(-0.5, 1)
plt.show()
'''
