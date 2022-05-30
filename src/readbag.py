#!/usr/bin/python

import rosbag
import matplotlib.pyplot as plt

bag = rosbag.Bag('crlb_based.bag')
crlb_covariance=[]
crlb_p1_error=[]
crlb_time = []

for topic, msg, t in bag.read_messages(topics='det_covariance'):

	crlb_covariance.append(msg.data[0])
	crlb_p1_error.append(msg.data[1])
	crlb_time.append(float(t.secs) + float(t.nsecs)/10**9)
	
bag.close()
ini = crlb_time[0]
crlb_time[:] = [x - ini for x in crlb_time]
'''
bag = rosbag.Bag('worst_exp_success.bag')
worst_covariance=[]
worst_p1_error=[]
worst_time = []

for topic, msg, t in bag.read_messages(topics='det_covariance'):

	worst_covariance.append(msg.data[0])
	worst_p1_error.append(msg.data[1])
	worst_time.append(float(t.secs) + float(t.nsecs)/10**9)

bag.close()
ini = worst_time[0]
worst_time[:] = [x - ini for x in worst_time]
'''
crlb_p1_avg_error = sum(crlb_p1_error[20:])/(len(crlb_p1_error)-20)
print(crlb_p1_avg_error)
crlb_p1_avg_error = [crlb_p1_avg_error]*len(crlb_p1_error)
'''
worst_p1_avg_error = sum(worst_p1_error[20:])/(len(worst_p1_error)-20)
print(worst_p1_avg_error)
worst_p1_avg_error = [worst_p1_avg_error]*len(worst_p1_error)
'''
crlb_avg_covariance = sum(crlb_covariance[20:])/(len(crlb_covariance)-20)
print(crlb_avg_covariance)
'''
worst_avg_covariance = sum(worst_covariance[20:])/(len(worst_covariance)-20)
print(worst_avg_covariance)
'''

plt.plot(crlb_time,crlb_covariance,"r",label="crlb_based")
#plt.plot(worst_time,worst_covariance,"b",label="worst")
plt.ylabel('det(P)')
plt.xlabel('time')
plt.ylim(1e-11, 1e-7)
plt.xlim(-0.5, 75)
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
#plt.plot(worst_time,worst_p1_error,"b",label="worst")
#plt.plot(worst_time,worst_p1_avg_error,"k",linestyle="dotted",label="worst_mean")
plt.ylabel('p1_error')
plt.xlabel('time')
plt.ylim(0, 0.5)
plt.xlim(-0.5, 75)
plt.legend()
plt.show()
'''
plt.plot(optimal_time,optimal_p1_error,"r")
plt.plot(worst_time,worst_p1_error,"b")
plt.ylim(0, 0.1)
plt.xlim(-0.5, 1)
plt.show()
'''
