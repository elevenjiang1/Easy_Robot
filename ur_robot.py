import os
import sys
import urx
import time
import cv2 as cv
import numpy as np

import signal
import threading
#For canceling signal
signal.signal(signal.SIGINT, quit)
signal.signal(signal.SIGTERM, quit)



class UR_Robot:
    def __init__(self,robot_ip):
        self.robot_ip=robot_ip
        self._robot=urx.Robot(self.robot_ip)
        
        self._q=None
        self._x=None
        
        
        robot_state_thread=threading.Thread(target=self.robot_state_cb)
        robot_state_thread.start()
        
        while self._q is None:
            print("Wait for robots joints...")
            time.sleep(1)
            
        print("UR robot: {} is ready!".format(self.robot_ip))
        
        
    def robot_state_cb(self):
        while True:
            self._q=self._robot._get_joints_dist()
            self._x=self._robot.get_pose()
            
            time.sleep(0.008)#Set in 125hz
        
    def get_robot_state(self):
        return self._q,self._x
    
    
def read_images():
    camera=cv.VideoCapture(0)
    while True:
        begin_time=time.time()
        ret,frame=camera.read()
        
        cv.imshow("image",frame)
        
        print("camera hz is:{}".format(1/(time.time()-begin_time)))
        
        key_input=cv.waitKey(1)
        if key_input==ord('q'):
            break
        
def basic_get_robot():
    robot=UR_Robot(robot_ip="192.168.0.1")
    robot_joints,robot_pose=robot.get_robot_state()
    print(robot_joints)
    print(robot_pose)
    




if __name__ == "__main__":
    read_images()
    # basic_get_robot()
