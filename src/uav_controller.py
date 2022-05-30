#!/usr/bin/python

import rospy
from math import sin,cos,sqrt,atan2,acos,pi
import numpy as np
import threading
from gazebo_msgs.msg import ModelStates
from scipy.optimize import minimize
from geometry_msgs.msg import Twist
from px4_mavros import Px4Controller
from std_msgs.msg import Float64MultiArray
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.factory import get_sampling, get_crossover, get_mutation, get_termination
from pymoo.optimize import minimize

P1,Pc,thetac,A,b = None,None,None,None,None
camera_cmd_vel = Twist()
#fx,fy,lx,ly = 0.1496485702,0.1496485702,0.1693333333,0.127
fx,fy,lx,ly = 565.6,565.6,640,480
sigma_u,sigma_v = 0.007,0.007
#sigma_u,sigma_v = 1,1
x_fov_wealth = 3*pi/180
y_fov_wealth = 3*pi/180
height_l = 0.3
height_u = 100
d_safe_car = 0.7
d_measuring = 2.2
gamma = 1.0
time_last,dt = 0,0

class Objective(ElementwiseProblem):

	def __init__(self):
		super().__init__(n_var=4,n_obj=2,n_constr=b.size, \
						 xl=np.array([-0.5]*4),xu=np.array([0.5]*4))
		self.cons = []

	def _evaluate(self, x, out, *args, **kwargs):
		
		nc = np.array([cos(thetac + dt*x[3]),sin(thetac + dt*x[3]),0])
		r1c_xy = np.array([P1[0] - (Pc[0] + dt*x[0]),P1[1] - (Pc[1] + dt*x[1]),0])
		r1c_z = np.array([0,0,P1[2] - Pc[2]])

		f1 = (np.linalg.norm(r1c_z))**2
		f2 = (nc.dot(r1c_xy))**6

		for i in range (b.size):
			self.cons += list(A[i,0]*x[0] + A[i,1]*x[1] + A[i,2]*x[2] + A[i,3]*x[3] - b[i])

		out["F"] = [f2/f1]
		out["G"] = self.cons
		self.cons = []

def odom(msg):
	global P1,Pc,A,b,thetac

	#UAV1_index = msg.name.index('iris_camera')
	#car1_index = msg.name.index('car1')
	#Pc = np.array([msg.pose[UAV1_index].position.x, msg.pose[UAV1_index].position.y, msg.pose[UAV1_index].position.z])
	#P1 = np.array([msg.pose[car1_index].position.x, msg.pose[car1_index].position.y, msg.pose[car1_index].position.z])
	P1 = np.array([msg.data[0], msg.data[1], 0])
	Pc = np.array([msg.data[4], msg.data[5], msg.data[6]])
	thetac = msg.data[7]
	
	nc = np.array([cos(thetac),sin(thetac),0])
	nc_dot = np.array([-sin(thetac),cos(thetac),0])
	r1c_xy = np.array([P1[0] - Pc[0],P1[1] - Pc[1],0])
	r1c_z = np.array([0,0,P1[2] - Pc[2]])

	'''
	[-2*(Pc[0]-P1[0]), -2*(Pc[1]-P1[1])]+[0]*2, \
	[2*(Pc[0]-P1[0]), 2*(Pc[1]-P1[1])]+[0]*2, \
	'''
	A = np.array([ \
				  np.append(-(np.dot(nc,r1c_xy)*r1c_xy/np.linalg.norm(r1c_xy)**3-nc/np.linalg.norm(r1c_xy))/sqrt(1 - np.dot(nc,r1c_xy)**2/np.linalg.norm(r1c_xy)**2),-np.dot(nc_dot,r1c_xy)/np.linalg.norm(r1c_xy)/sqrt(1 - np.dot(nc,r1c_xy)**2/np.linalg.norm(r1c_xy)**2)), \
				  np.append((np.linalg.norm(r1c_z)*nc/np.dot(nc,r1c_xy)**2-r1c_z/np.linalg.norm(r1c_z)/np.dot(nc,r1c_xy))/(1 + np.linalg.norm(r1c_z)**2/np.dot(nc,r1c_xy)**2),-np.linalg.norm(r1c_z)*np.dot(nc_dot,r1c_xy)/np.dot(nc,r1c_xy)**2/(1 + np.linalg.norm(r1c_z)**2/np.dot(nc,r1c_xy)**2)), \
				  [0]*2+[-1]+[0], \
				  ])

	'''
	[np.linalg.norm([Pc[0]-P1[0],Pc[1]-P1[1]])**2 - d_safe_car**2], \
	[d_measuring**2 - np.linalg.norm([Pc[0]-P1[0],Pc[1]-P1[1]])**2], \
	'''
	b = np.array([ \
				  [atan2(lx,2*fx) - x_fov_wealth - acos(np.dot(nc,r1c_xy)/np.linalg.norm(r1c_xy))], \
				  [atan2(ly,2*fy) - y_fov_wealth - atan2(np.linalg.norm(r1c_z),np.dot(nc,r1c_xy))], \
				  [Pc[2] - height_l], \
				  ])

def	qpsolver():
	global camera_cmd_vel,time_last,dt
	print((P1[2] - Pc[2])**2/(cos(thetac)*(P1[0] - Pc[0]) + sin(thetac)*(P1[1] - Pc[1]))**6)
	objective = Objective()
	algorithm = NSGA2(pop_size=20,n_offsprings=None,sampling=get_sampling("real_random"), \
					  crossover=get_crossover("real_sbx", prob=0.9, eta=15), \
					  mutation=get_mutation("real_pm", eta=20), eliminate_duplicates=True)
	termination = get_termination("n_gen", 8)
	dt = rospy.Time.now().to_sec() - time_last
	res = minimize(objective, algorithm, termination, seed=1, save_history=True, verbose=False, return_least_infeasible=False)
	
	tmp = np.inf
	num_opt = 0

	for i in range(len(res.F[:,0])):
		if np.prod(res.F[i,:]) < tmp:
			tmp = np.prod(res.F[i,:])
			num_opt = i

	optimal = res.X[num_opt,:4]
	#print(optimal)
	
	camera_cmd_vel.linear.x = optimal[0]
	camera_cmd_vel.linear.y = optimal[1]
	camera_cmd_vel.linear.z = optimal[2]
	camera_cmd_vel.angular.z = optimal[3]
	
	px4_camera.vel_control(camera_cmd_vel)
	time_last = rospy.Time.now().to_sec()
	
if __name__ == '__main__':
	try:
		rospy.init_node('controller')
		uavtype = ["iris_camera"]
		px4_camera = Px4Controller(uavtype[0])
		rate = rospy.Rate(50)
		
		#while thetac == None:
			#thetac = px4_camera.current_heading

		while not rospy.is_shutdown():
			msg = rospy.wait_for_message('/state', Float64MultiArray)
			#thetac = px4_camera.current_heading		
			odom(msg)
		
			qpsolver()
			rate.sleep()
	except rospy.ROSInterruptException:
		pass
