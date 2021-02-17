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

        #init q matrix
        self.q_matrix = [[0 for i in range(8)] for j in range(64)] # q.matrix[i] = the index in the action matrix of optimal state 
        
        # define red = 0, green = 1, blue = 2
        # define origin = 0, block 1 = 1, ....
        self.actions = []
        # let actions[i] = (location red, location green, location blue) in state i
        for blue_loc in range(4):
            for green_loc in range(4):
                for red_loc in range(4):
                    self.actions.append((red_loc, green_loc, blue_loc))

         # creating the action matrix
        self.action_matrix = [[-1 for i in range(64)] for j in range(64)]
        for start in self.actions:
            for goal in self.actions:
                index1 = self.actions.index(start)
                index2 = self.actions.index(goal)

                # check that goal state is valid
                for i in range(3):
                    for j in range(3):
                        if goal[i] == goal[j] and goal[i] != 0:
                            self.action_matrix == -1
                            continue
                
                #check only one dumbell is moving at a time
                diff = (start[0] != goal[0]) + (start[1] != goal[1]) + (start[2] != goal[2])
                if diff == 1:
                    for i in range(3):
                        if start[i] != goal[i]:
                            if goal[i] != 0:                 
                                action = goal[i] - 1
                                color = i
                                action += color*3
                                self.action_matrix[index1][index2] = action

                    

    def q_algorithm(self):
        # to do
        while self.counter < 20:
            print("Executing algorithm")

    def get_opt(self, state):
        #get the optimal action for a given state 
        opt = max(self.q_matrix[state])
        return self.q_matrix.index(opt)

if __name__ == "__main__":
    Q = QLearning()
