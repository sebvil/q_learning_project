#!/usr/bin/env python3
"""Tests for q_learning.py."""

import unittest

import q_learning
from q_learning_project.msg import QLearningReward


class TestParticleFilter(unittest.TestCase):
    def setUp(self):
        self.q_learning = q_learning.QLearning()
        pass

    def test_update_q_matrix(self):
        # first move
        self.q_learning.action = 0
        self.q_learning.update_q_matrix(QLearningReward(reward=1))
        self.assertEqual(self.q_learning.q_matrix[0][0], 1)
        self.assertEqual(self.q_learning.state, 1)
        self.assertEqual(self.q_learning.counter, 0)

        # second move
        self.q_learning.action = 4
        self.q_learning.update_q_matrix(QLearningReward(reward=0.5))
        self.assertEqual(self.q_learning.q_matrix[1][4], 0.5)
        self.assertEqual(self.q_learning.state, 9)
        self.assertEqual(self.q_learning.counter, 0)

        # third move
        self.q_learning.state = 0
        self.q_learning.action = 0
        self.q_learning.update_q_matrix(QLearningReward(reward=1))
        self.assertEqual(self.q_learning.q_matrix[0][0], 1.75)
        self.assertEqual(self.q_learning.state, 1)
        self.assertEqual(self.q_learning.counter, 0)

        # fourth move
        self.q_learning.state = 0
        self.q_learning.action = 0
        self.q_learning.update_q_matrix(QLearningReward(reward=0.625))
        self.assertEqual(self.q_learning.q_matrix[0][0], 1.75)
        self.assertEqual(self.q_learning.state, 1)
        self.assertEqual(self.q_learning.counter, 1)


if __name__ == "__main__":
    unittest.main()
