#!/usr/bin/env python3
"""Tests for q_learning.py."""

import unittest
from unittest.mock import Mock
import q_learning
from q_learning_project.msg import QLearningReward


class TestQLearning(unittest.TestCase):
    def setUp(self):
        self.q_learning = q_learning.QLearning(test_mode=True)
        self.q_learning.q_matrix_publisher = Mock()
        pass

    def test_is_goal_valid(self):
        self.assertTrue(self.q_learning._is_goal_valid((0, 0, 0)))
        self.assertTrue(self.q_learning._is_goal_valid((0, 0, 1)))
        self.assertTrue(self.q_learning._is_goal_valid((0, 2, 1)))
        self.assertTrue(self.q_learning._is_goal_valid((3, 2, 1)))
        self.assertTrue(self.q_learning._is_goal_valid((2, 3, 1)))

        self.assertFalse(self.q_learning._is_goal_valid((0, 1, 1)))
        self.assertFalse(self.q_learning._is_goal_valid((1, 0, 1)))
        self.assertFalse(self.q_learning._is_goal_valid((2, 2, 1)))
        self.assertFalse(self.q_learning._is_goal_valid((3, 2, 3)))
        self.assertFalse(self.q_learning._is_goal_valid((2, 3, 3)))

    def test_is_move_valid(self):
        self.assertTrue(self.q_learning._is_move_valid((0, 0, 0), (0, 0, 1)))
        self.assertTrue(self.q_learning._is_move_valid((0, 0, 1), (2, 0, 1)))
        self.assertTrue(self.q_learning._is_move_valid((2, 0, 1), (2, 3, 1)))

        self.assertFalse(self.q_learning._is_move_valid((0, 0, 1), (0, 0, 1)))
        self.assertFalse(self.q_learning._is_move_valid((0, 0, 1), (2, 3, 1)))
        self.assertFalse(self.q_learning._is_move_valid((2, 0, 1), (3, 0, 1)))
        self.assertFalse(self.q_learning._is_move_valid((0, 0, 0), (2, 1, 0)))

    def test_init(self):
        """Tests that the action and Q matrices are properly initialized."""
        q_matrix = self.q_learning.q_matrix
        action_matrix = self.q_learning.q_matrix
        self.assertEqual(len(q_matrix), 64)
        for row in q_matrix:
            self.assertEqual(len(row), 9)
            for val in row:
                self.assertEqual(val, 0)

        self.assertEqual(len(action_matrix), 64)

        for row_i, row in enumerate(self.q_learning.action_matrix):
            self.assertEqual(len(row), 64)
            for col_i, col in enumerate(row):
                start = (row_i % 4, (row_i // 4) % 4, (row_i // 16))
                goal = (col_i % 4, (col_i // 4) % 4, (col_i // 16))
                if not self.q_learning._is_goal_valid(goal):
                    self.assertEqual(col, -1)
                    continue

                if not self.q_learning._is_move_valid(start, goal):
                    self.assertEqual(col, -1)
                    continue

                dumbbell_index = col // 3
                block_number = col % 3 + 1
                i = 0
                for x, y in zip(start, goal):
                    if i != dumbbell_index:
                        self.assertEqual(x, y)
                    else:
                        self.assertEqual(y, block_number)
                    i += 1

    def test_update_q_matrix(self):
        self.q_learning.action_states_queue.append((1, 0))
        self.q_learning.action_states_queue.append((9, 4))
        self.q_learning.action_states_queue.append((57, 8))
        self.q_learning.action_states_queue.append((1, 0))

        # first move
        self.q_learning.update_q_matrix(QLearningReward(reward=1))
        self.assertEqual(self.q_learning.q_matrix[0][0], 1)
        self.assertEqual(self.q_learning.counter, 0)

        # second move
        self.q_learning.update_q_matrix(QLearningReward(reward=0.5))
        self.assertEqual(self.q_learning.q_matrix[1][4], 0.5)
        self.assertEqual(self.q_learning.counter, 0)

        # third move
        self.q_learning.action = 8
        self.q_learning.update_q_matrix(QLearningReward(reward=0.8))
        self.assertEqual(self.q_learning.q_matrix[9][8], 0.8)
        self.assertEqual(self.q_learning.counter, 0)

        # fourth move
        self.q_learning.state = 0
        self.q_learning.action = 0
        self.q_learning.update_q_matrix(QLearningReward(reward=0.25))
        self.assertEqual(self.q_learning.q_matrix[0][0], 1)
        self.assertEqual(self.q_learning.counter, 1)


if __name__ == "__main__":
    unittest.main()
