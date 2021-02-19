#!/usr/bin/env python3

import rospy

from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Quaternion, Point, Pose, PoseArray, PoseStamped
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Header, String
from q_learning_project.msg import QLearningReward, RobotMoveDBToBlock, QMatrix
import random
import time


class QLearning:
    def __init__(self, gamma=0.5, alpha=1):
        rospy.init_node("turtlebot3_q_learning")

        self.counter = 0  # check if q algorithm has converged
        self.gamma = gamma  # discount factor
        self.alpha = alpha  # learning rate
        self.epsilon = 0.001

        # init q matrix
        self.q_matrix = [
            [0] * 8 for j in range(64)
        ]  # q_matrix[i] = the index in the action matrix of optimal state

        # define red = 0, green = 1, blue = 2
        # define origin = 0, block 1 = 1, ....
        self.actions = []
        # let actions[i] = (location red, location green, location blue) in state i
        for blue_loc in range(4):
            for green_loc in range(4):
                for red_loc in range(4):
                    self.actions.append((red_loc, green_loc, blue_loc))

        # creating the action matrix
        self.action_matrix = [[-1] * 64 for j in range(64)]
        for index1, start in enumerate(self.actions):
            for index2, goal in enumerate(self.actions):
                # check that goal state is valid
                goal_valid = True
                found_pos = set()
                for position in goal:
                    if position != 0 and position in found_pos:
                        goal_valid = False
                        break
                    found_pos.add(position)
                if not goal_valid:
                    continue
                # check only one dumbell is moving at a time
                diff = (
                    (start[0] != goal[0])
                    + (start[1] != goal[1])
                    + (start[2] != goal[2])
                )
                if diff == 1:
                    for color in range(3):
                        if start[color] != goal[color] and goal[color] != 0:
                            action = goal[color] - 1 + color * 3
                            self.action_matrix[index1][index2] = action

        self.reward_subscriber = rospy.Subscriber(
            "/q_learning/reward", QLearningReward, self.update_q_matrix
        )
        self.move_publisher = rospy.Publisher(
            "/q_learning/robot_action", RobotMoveDBToBlock, queue_size=10
        )
        self.q_matrix_publisher = rospy.Publisher(
            "/q_learning/q_matrix", QMatrix, queue_size=10
        )

        self.q_matrix_publisher.publish(QMatrix(q_matrix=self.q_matrix))
        self.index_color_map = {0: "red", 1: "green", 2: "blue"}
        # ints representing the current state and action taken
        self.action = None
        self.state = 0
        self.waiting_for_reward = False

    def update_q_matrix(self, data):
        """Updates thr Q-matrix based on the give reward."""
        print("received")
        reward = data.reward
        next_state = self.action_matrix[self.state].index(self.action)
        next_actions_diffs = [
            x - self.q_matrix[self.state][self.action]
            for x in self.q_matrix[next_state]
        ]
        new_value = self.q_matrix[self.state][self.action] + self.alpha * (
            reward + self.gamma * max(next_actions_diffs)
        )
        if (
            abs(new_value - self.q_matrix[self.state][self.action])
            < self.epsilon
        ):
            self.counter += 1
        else:
            self.counter = 0
        self.q_matrix[self.state][self.action] = new_value
        self.state = next_state
        self.q_matrix_publisher.publish(QMatrix(q_matrix=self.q_matrix))

        self.waiting_for_reward = False

    def q_algorithm(self):
        # to do
        while self.counter < 20:
            if self.waiting_for_reward:
                continue
            possible_actions = [
                i for i in self.action_matrix[self.state] if i != -1
            ]
            self.action = random.choice(possible_actions)
            self.move_publisher.publish(
                RobotMoveDBToBlock(
                    robot_db=self.index_color_map[self.action // 3],
                    block_id=self.action % 3 + 1,
                )
            )
            # self.waiting_for_reward = True
            time.sleep(1)
            print("Executing algorithm")

    def get_opt(self, state):
        # get the optimal action for a given state
        opt = max(self.q_matrix[state])
        return self.q_matrix.index(opt)

    def run(self):
        self.q_algorithm()


if __name__ == "__main__":
    QLearning().run()
