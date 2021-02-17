#!/usr/bin/env python3

import rospy

from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Quaternion, Point, Pose, PoseArray, PoseStamped
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Header, String

import random

class QLearning:

    def __init__(self):
        self.counter = 0 # check if q algorithm has converged

        #self.q_matrix = [0]*63 # q.matrix[i] = the index in the action matrix of optimal state 
        
        
        # define red = 0, green = 1, blue = 2
        # creating the action matrix

        ## this stuff is not done
        self.action_matrix = [[0]*63]*63
        for begin in range(63):
            for end in range(63):
                if begin == end:
                    self.action_matrix[begin][end] = -1
                
                new_begin = begin
                new_end = end
                while new_begin > 15:
                    new_begin -= 16
                while new_end > 15:
                    new_end -= 16     
                # check if the end state is divisible by 5 (these are invalid)
                if new_end != 0 and (new_end % 5 == 0):
                    self.action_matrix[begin][end] = -1
                else:
                    self.action_matrix[begin][end] = 
                     
                
            



    def q_algorithm(self):
        # to do
        while self.counter < 20:
            print("Executing algorithm")

    def get_opt(self, state):
        #get the optimal action for a given state (state = index of q matrix)
        max = max(self.q_matrix[state])
        return self.q_matrix.index(max)

if __name__ == "__main__":
    Q = QLearning()