#!/usr/bin/env python

import rospy
import numpy as np
from geometry_msgs.msg import Twist

car1_cmd_vel = Twist()
time = 0

def start():
	global car1_cmd_vel,time
	car1_cmd_vel.linear.x = 0.1
	car1_vel_pub.publish(car1_cmd_vel)
	time = 1

def control():
	global car1_cmd_vel,time
	
	if time < 151:
		car1_cmd_vel.linear.x = 0.2
	elif time < 301:
		car1_cmd_vel.linear.x = 0.2
		car1_cmd_vel.angular.z = -0.4
	elif time < 376:
		car1_cmd_vel.linear.x = 0.2
		car1_cmd_vel.angular.z = 0.0
	elif time < 526:
		car1_cmd_vel.linear.x = 0.2
		car1_cmd_vel.angular.z = 0.4
	elif time < 601:
		car1_cmd_vel.linear.x = 0.2
		car1_cmd_vel.angular.z = 0.0
	
	car1_vel_pub.publish(car1_cmd_vel)
	time = time+1

def stop():
	global car1_cmd_vel
	car1_cmd_vel.linear.x = 0.0
	car1_cmd_vel.angular.z = 0.0
	car1_vel_pub.publish(car1_cmd_vel)

if __name__ == '__main__':
	try:
		rospy.init_node('navigation')
		car1_vel_pub = rospy.Publisher("/car1/cmd_vel",Twist,queue_size=1)
		rate = rospy.Rate(20)
		while not rospy.is_shutdown():
			if time == 0:
				start()
			elif time < 601:
				control()
			else:
				stop()
				break
			rate.sleep()
	except rospy.ROSInterruptException:
		pass
