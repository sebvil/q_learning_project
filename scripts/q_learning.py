#!/usr/bin/env python3

import collections
import random
import numpy as np
import rospy
from geometry_msgs.msg import Twist
from q_learning_project.msg import (
    QLearningReward,
    QMatrix,
    RobotMoveDBToBlock,
    QMatrixRow,
)
from tf.transformations import euler_from_quaternion


def get_yaw_from_pose(p):
    """ A helper function that takes in a Pose object (geometry_msgs) and returns yaw"""

    yaw = euler_from_quaternion(
        [p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w]
    )[2]

    return yaw


class QLearning:
    def __init__(self, gamma=0.5, alpha=1, test_mode=False):
        self.counter = 0  # check if q algorithm has converged
        self.gamma = gamma  # discount factor
        self.alpha = alpha  # learning rate
        self.epsilon = 1

        # init q matrix
        self.q_matrix = QMatrix(q_matrix=[])
        for i in range(64):
            # q_matrix[i] = the index in the action matrix of optimal state
            self.q_matrix.q_matrix.append(QMatrixRow(q_matrix_row=[0] * 9))

        self._init_action_matrix()
        self.actions_sequence = []
        for i, action_1 in enumerate(self.action_matrix[0]):
            if action_1 == -1:
                continue
            for j, action_2 in enumerate(self.action_matrix[i]):
                if action_2 == -1:
                    continue
                for k, action_3 in enumerate(self.action_matrix[j]):
                    if action_3 == -1:
                        continue
                    self.actions_sequence.extend(
                        [action_1, action_2, action_3]
                    )

        if not test_mode:
            rospy.init_node("turtlebot3_q_learning", anonymous=True)

            self.reward_subscriber = rospy.Subscriber(
                "/q_learning/reward", QLearningReward, self.update_q_matrix
            )

            self.move_publisher = rospy.Publisher(
                "/q_learning/robot_action", RobotMoveDBToBlock, queue_size=10
            )
            self.q_matrix_publisher = rospy.Publisher(
                "/q_learning/q_matrix", QMatrix, queue_size=10, latch=True
            )

            self.q_matrix_publisher.publish(self.q_matrix)

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
        # let actions[i] = (location red, location green, location blue) in
        #  state i
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

    def _get_q_row(self, state):
        return self.q_matrix.q_matrix[state].q_matrix_row

    def _get_q_cell(self, state, action):
        return self.q_matrix.q_matrix[state].q_matrix_row[action]

    def _set_q_cell(self, state, action, value):
        self.q_matrix.q_matrix[state].q_matrix_row[action] = int(value)

    def update_q_matrix(self, data):
        """Updates thr Q-matrix based on the give reward."""
        print(data)
        if not self.action_states_queue:
            return
        reward = data.reward

        state, next_state, action = self.action_states_queue.popleft()

        current_value = self._get_q_cell(state, action)
        next_state_row = self._get_q_row(next_state)
        new_value = current_value + self.alpha * (
            reward + self.gamma * max(next_state_row) - current_value
        )

        if abs(new_value - current_value) < self.epsilon:
            self.counter += 1
        else:
            self.counter = 0
        self._set_q_cell(state, action, new_value)

        self.q_matrix_publisher.publish(self.q_matrix)

    def q_algorithm(self):
        actions = [2, 3, 7]
        while (
            sum(self._get_q_row(0)) != self.gamma ** 2 * 100 or self.state != 0
        ):
            rospy.sleep(1.5)
            print(self.iterations, self.state, self._get_q_row(self.state))
            possible_actions = [
                i for i in self.action_matrix[self.state] if i != -1
            ]
            action = actions[
                self.iterations % 3
            ]  # random.choice(possible_actions)
            next_state = self.action_matrix[self.state].index(action)
            self.action_states_queue.append((self.state, next_state, action))
            self.iterations += 1

            self.move_publisher.publish(
                RobotMoveDBToBlock(
                    robot_db=self.index_color_map[action // 3],
                    block_id=action % 3 + 1,
                )
            )
            if self.iterations % 3 == 0:
                self.state = 0
            else:
                if next_state == 0:
                    print(self.state, action, next_state)
                self.state = next_state
        self.converged = True
        print("Converged. Close phantom_movement.py")

    def run(self):
        self.q_algorithm()

        # rospy.sleep(5)  # gazebo takes a second to get started

        # # retrieving both theta values and (x,y) tuples for blocks and db for
        # # more options
        # self.controller.find_db_order()
        # self.controller.find_db_locs()

        # self.controller.turn(np.pi)
        # rospy.sleep(1)
        # self.controller.find_block_thetas(self.ranges)
        # self.controller.find_block_order()
        # self.controller.turn(0)
        # self.controller.converged = True

        # self.q_matrix_publisher.publish(QMatrix(q_matrix=self.q_matrix))
        rospy.spin()


if __name__ == "__main__":
    Q = QLearning()

    Q.run()
