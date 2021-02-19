#!/usr/bin/env python3

import rospy

from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Quaternion, Point, Pose, PoseArray, PoseStamped
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Header, String
from q_learning_project.msg import QLearningReward, RobotMoveDBToBlock, QMatrix
import random
import time
import collections


class QLearning:
    def __init__(self, gamma=0.5, alpha=1, test_mode=False):

        self.counter = 0  # check if q algorithm has converged
        self.gamma = gamma  # discount factor
        self.alpha = alpha  # learning rate
        self.epsilon = 1

        # init q matrix
        self.q_matrix = [
            [0] * 9 for j in range(64)
        ]  # q_matrix[i] = the index in the action matrix of optimal state

        self._init_action_matrix()
        if not test_mode:
            rospy.init_node("turtlebot3_q_learning")

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
        self.action_states_queue = collections.deque()
        self.state = 0
        self.iterations = 0
        self.waiting_for_reward = False

    def _init_action_matrix(self):
        # define red = 0, green = 1, blue = 2
        # define origin = 0, block 1 = 1, ....
        actions = []
        # let actions[i] = (location red, location green, location blue) in state i
        for blue_loc in range(4):
            for green_loc in range(4):
                for red_loc in range(4):
                    actions.append((red_loc, green_loc, blue_loc))

        # creating the action matrix
        self.action_matrix = [[-1] * 64 for j in range(64)]
        for index1, start in enumerate(actions):
            for index2, goal in enumerate(actions):
                # check that goal state is valid
                goal_valid = self._is_goal_valid(goal)
                if not goal_valid:
                    continue
                # check only one dumbell is moving at a time

                if self._is_move_valid(start, goal):
                    for color in range(3):
                        if start[color] < goal[color] and goal[color] != 0:
                            action = goal[color] - 1 + color * 3
                            self.action_matrix[index1][index2] = action

    def _is_goal_valid(self, goal):
        goal_valid = True
        found_pos = set()
        for position in goal:
            if position != 0 and position in found_pos:
                goal_valid = False
                break
            found_pos.add(position)
        return goal_valid

    def _is_move_valid(self, start, goal):
        moves = 0
        for x, y in zip(start, goal):
            if y != x and x == 0:
                moves += 1
            elif y != x:
                return False
        return moves == 1

    def update_q_matrix(self, data):
        """Updates thr Q-matrix based on the give reward."""
        if not self.action_states_queue:
            return
        reward = data.reward
        state, next_state, action = self.action_states_queue.popleft()
        next_actions_diffs = [
            x - self.q_matrix[state][action] for x in self.q_matrix[next_state]
        ]
        new_value = self.q_matrix[state][action] + self.alpha * (
            reward + self.gamma * max(next_actions_diffs)
        )
        if abs(new_value - self.q_matrix[state][action]) < self.epsilon:
            print(
                "new: {}, old: {}, state: {}, counter: {}, reward: {}".format(
                    new_value,
                    self.q_matrix[state][action],
                    state,
                    self.counter,
                    data.iteration_num,
                )
            )
            self.counter += 1
        else:
            self.counter = 0
        self.q_matrix[state][action] = new_value

        self.q_matrix_publisher.publish(QMatrix(q_matrix=self.q_matrix))
        self.waiting_for_reward = False

    def q_algorithm(self):
        # to do
        self.last_action = -1
        while self.counter < 20:
            print(self.counter, self.iterations)
            time.sleep(0.5)
            possible_actions = [
                i for i in self.action_matrix[self.state] if i != -1
            ]
            action = random.choice(possible_actions)
            next_state = self.action_matrix[self.state].index(action)
            self.action_states_queue.append((self.state, next_state, action))
            self.iterations += 1

            self.move_publisher.publish(
                RobotMoveDBToBlock(
                    robot_db=self.index_color_map[action // 3],
                    block_id=action % 3 + 1,
                )
            )
            if self.last_action == action:
                print("ACTION REPEATED", self.state, next_state, action)
            self.last_action = action
            if self.iterations % 3 == 0:
                self.state = 0
            else:
                self.state = next_state

    def get_opt(self, state):
        # get the optimal action for a given state
        opt = max(self.q_matrix[state])
        return self.q_matrix.index(opt)

    def run(self):
        self.q_algorithm()


if __name__ == "__main__":
    QLearning().run()
