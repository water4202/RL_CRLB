#!/usr/bin/env python

import rospy
import rosnode
import numpy as np
from geometry_msgs.msg import Twist

class CarTrain:
    def __init__(self):
        self.car1_vel_pub = rospy.Publisher("/car1/cmd_vel",Twist,queue_size=1)
        self.rate = rospy.Rate(20)

        self.car1_cmd_vel = Twist()
        self.time = 0

        rospy.set_param("/car_navigation/start", 0)

    def start(self):
        self.vx = np.random.uniform(0, 0.2, 6)
        self.vz = np.random.uniform(-0.4, 0.4, 6)
        self.car1_cmd_vel.linear.x = self.vx[0]
        self.car1_cmd_vel.angular.z = self.vz[0]
        self.car1_vel_pub.publish(self.car1_cmd_vel)
        self.time = 1

    def control(self):
        if self.time < 151:
            self.car1_cmd_vel.linear.x = self.vx[1]
            self.car1_cmd_vel.angular.z = self.vz[1]
        elif self.time < 301:
            self.car1_cmd_vel.linear.x = self.vx[2]
            self.car1_cmd_vel.angular.z = self.vz[2]
        elif self.time < 376:
            self.car1_cmd_vel.linear.x = self.vx[3]
            self.car1_cmd_vel.angular.z = self.vz[3]
        elif self.time < 526:
            self.car1_cmd_vel.linear.x = self.vx[4]
            self.car1_cmd_vel.angular.z = self.vz[4]
        elif self.time < 601:
            self.car1_cmd_vel.linear.x = self.vx[5]
            self.car1_cmd_vel.angular.z = self.vz[5]

        self.car1_vel_pub.publish(self.car1_cmd_vel)
        self.time = self.time+1

    def stop(self):
        self.car1_cmd_vel.linear.x = 0.0
        self.car1_cmd_vel.angular.z = 0.0
        self.car1_vel_pub.publish(self.car1_cmd_vel)
        self.time = self.time+1

class CarTest:
    def __init__(self):
        self.car1_vel_pub = rospy.Publisher("/car1/cmd_vel",Twist,queue_size=1)
        self.rate = rospy.Rate(20)

        self.car1_cmd_vel = Twist()
        self.time = 0

        rospy.set_param("/car_navigation/start", 0)

    def start(self):
        self.car1_cmd_vel.linear.x = 0.1
        self.car1_cmd_vel.angular.z = 0.0
        self.car1_vel_pub.publish(self.car1_cmd_vel)
        self.time = 1

    def control(self):
        if self.time < 151:
            self.car1_cmd_vel.linear.x = 0.2
            self.car1_cmd_vel.angular.z = 0.0
        elif self.time < 301:
            self.car1_cmd_vel.linear.x = 0.2
            self.car1_cmd_vel.angular.z = -0.4
        elif self.time < 376:
            self.car1_cmd_vel.linear.x = 0.2
            self.car1_cmd_vel.angular.z = 0.0
        elif self.time < 526:
            self.car1_cmd_vel.linear.x = 0.2
            self.car1_cmd_vel.angular.z = 0.4
        elif self.time < 601:
            self.car1_cmd_vel.linear.x = 0.2
            self.car1_cmd_vel.angular.z = 0.0

        self.car1_vel_pub.publish(self.car1_cmd_vel)
        self.time = self.time+1

    def stop(self):
        self.car1_cmd_vel.linear.x = 0.0
        self.car1_cmd_vel.angular.z = 0.0
        self.car1_vel_pub.publish(self.car1_cmd_vel)
        self.time = self.time+1

if __name__ == '__main__':
    try:
        rospy.init_node('navigation')
        car = None

        while car is None:
            if rosnode.rosnode_ping('/train_node', max_count=1) is True:
                car = CarTrain()
            if rosnode.rosnode_ping('/test_node', max_count=1) is True:
                car = CarTest()
            if rosnode.rosnode_ping('/controller', max_count=1) is True and rosnode.rosnode_ping('/estimate', max_count=1) is True:
                car = CarTest()
                rospy.set_param("/car_navigation/start", 1)

        while not rospy.is_shutdown():
            if rospy.get_param("/car_navigation/start") == 1:
                if car.time == 0:
                    car.start()
                elif car.time < 601:
                    car.control()
                elif car.time < 701:
                    car.stop()
                else:
                    car.time = 0
                    rospy.set_param("/car_navigation/start", 0)
            car.rate.sleep()
    except rospy.ROSInterruptException:
        pass
