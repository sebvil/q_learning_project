#!/usr/bin/env python3

import collections
import random

import rospy
from geometry_msgs.msg import Pose
from q_learning_project.msg import (
    QLearningReward,
    QMatrix,
    QMatrixRow,
    RobotMoveDBToBlock,
)
from tf.transformations import euler_from_quaternion


def get_yaw_from_pose(p: Pose) -> float:
    """Takes in a Pose object and returns yaw."""

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

        self.index_color_map = {0: "red", 1: "green", 2: "blue"}

        # ints representing the current state, next state, and action taken
        self.action_states_queue = collections.deque()
        self.state = 0
        self.iterations = 0

        # Init ROS node, publishers and subscribers if not testing.
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

    def _init_action_matrix(self):
        """Initializes the action matrix."""
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

                # If the move is valid, set the correct action at
                # (index1, index2)
                if self._is_move_valid(start, goal):
                    for color in range(3):
                        if start[color] < goal[color] and goal[color] != 0:
                            action = goal[color] - 1 + color * 3
                            self.action_matrix[index1][index2] = action

    def _is_goal_valid(self, goal: tuple[int, int, int]) -> bool:
        """Checks that the goal is a valid state.

        Parameters:
            goal: The locations of the red, green, and blue dumbbells, in that
            order.

        Returns:
            A boolean indicating if the state is valid, i.e., there is at most
            one dumbbell per block.
        """
        goal_valid = True
        found_pos = set()
        for position in goal:
            if position != 0 and position in found_pos:
                goal_valid = False
                break
            found_pos.add(position)
        return goal_valid

    def _is_move_valid(
        self, start: tuple[int, int, int], goal: tuple[int, int, int]
    ) -> bool:
        """Checks that the move from start to goal is valid.

        Parameters:
            start: The initial locations of the red, green, and blue dumbbells,
                in that order.
            goal: The end locations of the red, green, and blue dumbbells, in
                that order.

        Returns:
            A boolean indicating if the move from start to goal is valid, which
            means that only one dumbbell was moved and it was moved from the
            origin to a block.
        """
        moves = 0
        for x, y in zip(start, goal):
            if y != x and x == 0:
                moves += 1
            elif y != x:
                return False
        return moves == 1

    def _get_q_row(self, state: int) -> list[int]:
        """Returns the row of the Q-matrix corresponding to the given state."""
        return self.q_matrix.q_matrix[state].q_matrix_row

    def _get_q_cell(self, state: int, action: int) -> int:
        """Returns the value of the Q-matrix at the given state and action."""
        return self.q_matrix.q_matrix[state].q_matrix_row[action]

    def _set_q_cell(self, state: int, action: int, value: float) -> None:
        """Sets the value of the Q-matrix at the given state and action."""
        self.q_matrix.q_matrix[state].q_matrix_row[action] = int(value)

    def update_q_matrix(self, data):
        """Updates thr Q-matrix based on the received reward."""

        # if no actions have been taken, return
        if not self.action_states_queue:
            return

        reward = data.reward
        state, next_state, action = self.action_states_queue.popleft()

        current_value = self._get_q_cell(state, action)
        next_state_row = self._get_q_row(next_state)
        new_value = current_value + self.alpha * (
            reward + self.gamma * max(next_state_row) - current_value
        )

        # checks if the values are close enough (for convergence)
        if abs(new_value - current_value) < self.epsilon:
            self.counter += 1
        else:
            self.counter = 0

        # update Q-matrix and publish it
        self._set_q_cell(state, action, new_value)
        self.q_matrix_publisher.publish(self.q_matrix)

    def q_algorithm(self):
        """Selects and sends the actions taken by the Q-algorithm."""

        # Loop runs until the Q-matrix has converged and the state is back at 0
        while self.counter < 100 or self.state != 0:
            # Sleep to prevent race conditions.
            rospy.sleep(1.5)

            # Select an action at random.
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

            # Sets self.state to the correct value based on the number of
            # iterations
            if self.iterations % 3 == 0:
                self.state = 0
            else:
                self.state = next_state

        print("Converged. Close phantom_movement.py and run robot_control.py.")

    def run(self):
        self.q_algorithm()
        rospy.spin()


if __name__ == "__main__":
    QLearning().run()
