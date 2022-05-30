#!/usr/bin/python

import rospy
from math import sin,cos,sqrt,atan2,acos,pi
import numpy as np
import gurobipy as gp
from scipy.optimize import minimize
from geometry_msgs.msg import Twist
from px4_mavros import Px4Controller
from std_msgs.msg import Float64MultiArray

P1,Pc,thetac,A,b = None,None,None,None,None
camera_cmd_vel = Twist()
fx,fy,lx,ly = 565.6,565.6,640,480
sigma_u,sigma_v = 0.007,0.007
x_fov_wealth = 3*pi/180
y_fov_wealth = 3*pi/180
height_l = 0.3
height_u = 100
d_safe_car = 0.7
d_measuring = 2.2
gamma = 1.0
m,x = None,None

def odom(msg):
	global P1,Pc,A,b,thetac

	P1 = np.array([msg.data[0], msg.data[1], 0])
	Pc = np.array(msg.data[4:7])
	thetac = msg.data[7]
	
	nc = np.array([cos(thetac),sin(thetac),0])
	nc_dot = np.array([-sin(thetac),cos(thetac),0])
	r1c = P1-Pc

	A = np.array([ \
				  np.concatenate((-(nc.dot(r1c)*r1c[:2]/np.linalg.norm(r1c[:2])**3-nc[:2]/np.linalg.norm(r1c[:2]))/sqrt(1 - nc.dot(r1c)**2/np.linalg.norm(r1c[:2])**2),[0],[-nc_dot.dot(r1c)/np.linalg.norm(r1c[:2])/sqrt(1 - nc.dot(r1c)**2/np.linalg.norm(r1c[:2])**2)])), \
				  np.concatenate((abs(r1c[2])*nc[:2]/nc.dot(r1c)**2/(1 + r1c[2]**2/nc.dot(r1c)**2),[-r1c[2]/nc.dot(r1c)/abs(r1c[2])/(1 + r1c[2]**2/nc.dot(r1c)**2)],[-abs(r1c[2])*nc_dot.dot(r1c)/nc.dot(r1c)**2/(1 + r1c[2]**2/nc.dot(r1c)**2)])), \
				  [0]*2+[-1]+[0], \
				  [0]*2+[1]+[0] \
				  ])

				  #(-2*(Pc-P1)[:2]).tolist()+[0]*2, \
				  #(2*(Pc-P1)[:2]).tolist()+[0]*2, \

	b = np.array([ \
				  atan2(lx,2*fx) - x_fov_wealth - acos(nc.dot(r1c)/np.linalg.norm(r1c[:2])), \
				  atan2(ly,2*fy) - y_fov_wealth - atan2(abs(r1c[2]),nc.dot(r1c)), \
				  Pc[2] - height_l, \
				  height_u - Pc[2] \
				  ])*gamma

				  #np.linalg.norm((Pc-P1)[:2])**2 - d_safe_car**2, \
				  #d_measuring**2 - np.linalg.norm((Pc-P1)[:2])**2, \

def qp_ini():
	global m,x
	
	m = gp.Model("qp")
	m.setParam("NonConvex", 2.0)
	m.setParam("LogToConsole",0)
	x = m.addVars(4,ub=0.3, lb=-0.3, name="x")

def addCons(i):
	global m

	m.addConstr(A[i,0]*x[0] + A[i,1]*x[1] + A[i,2]*x[2] + A[i,3]*x[3] <= b[i], "c"+str(i))

def	qpsolver():
	global camera_cmd_vel,x

	obj = (x[0] - (P1 - Pc)[0])**2 + (x[1] - (P1 - Pc)[1])**2 - (x[2] - (P1 - Pc)[2])**2 - (thetac + x[3] - atan2((P1-Pc)[1],(P1-Pc)[0]))**2	# optimal

	m.setObjective(obj)

	m.remove(m.getConstrs())
	
	for i in range (b.size):
		addCons(i)

	m.optimize()
	optimal = m.getVars()
	#print(A.dot(np.array([optimal[0].X,optimal[1].X,optimal[2].X])) - b)
	
	camera_cmd_vel.linear.x = optimal[0].X
	camera_cmd_vel.linear.y = optimal[1].X
	camera_cmd_vel.linear.z = optimal[2].X
	camera_cmd_vel.angular.z = optimal[3].X
	
	px4_camera.vel_control(camera_cmd_vel)
	
if __name__ == '__main__':
	try:
		rospy.init_node('controller')
		px4_camera = Px4Controller("iris_camera")
		rospy.Subscriber('/state', Float64MultiArray, odom, queue_size=10)
		rate = rospy.Rate(100)
		while b is None:
			rate.sleep()

		qp_ini()
		while not rospy.is_shutdown():		
			qpsolver()
			rate.sleep()
	except rospy.ROSInterruptException:
		pass
