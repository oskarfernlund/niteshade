#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Unit and integration tests for the simulation module.
"""


# =============================================================================
#  IMPORTS AND DEPENDENCIES
# =============================================================================

import unittest

import pytest

from niteshade.attack import AddLabeledPointsAttacker, LabelFlipperAttacker, Attacker
from niteshade.defence import Defender, FeasibleSetDefender
from niteshade.model import IrisClassifier, MNISTClassifier
from niteshade.simulation import Simulator, wrap_results
from niteshade.utils import train_test_iris, train_test_MNIST


# =============================================================================
#  CLASSES
# =============================================================================

class TestDefender(Defender):
    def __init__(self, test):
        super.__init__(self)

    def defend(self):
        pass


class TestAttacker(Attacker):
    def __init__(self):
        super.__init__(self)
    
    def attack(self):
        pass

class Simulation_test(unittest.TestCase):
    def setUp(self) -> None:
        #data
        self.X_mnist_tr, self.y_mnist_tr, self.X_mnist_te, self.y_mnist_te = train_test_MNIST()
        self.X_iris_tr, self.y_iris_tr, self.X_iris_te, self.y_iris_te = train_test_iris()
        self.batch_size = 32
        self.num_episodes = 10

    def test_mnist(self):
        """Attack and defense combinations simulations for Iris classifier."""

        # Instantiate necessary classes
        # Instantiate necessary classes
        defender = FeasibleSetDefender(self.X_mnist_tr, self.y_mnist_tr, 2000)
        # defender = SoftmaxDefender(threshold=0.1)
        # attacker = SimpleAttacker(0.6, 1)
        
        dict = {1:4, 4:1, 3:5, 5:3}
        attacker = LabelFlipperAttacker(1, dict) 

        #implement attack and defense strategies through learner
        model = MNISTClassifier()
        simulator1 = Simulator(self.X_mnist_tr, self.y_mnist_tr, model, attacker=attacker,
                            defender=defender, batch_size=self.batch_size, num_episodes=self.num_episodes)

        model = MNISTClassifier()
        simulator2 = Simulator(self.X_mnist_tr, self.y_mnist_tr, model, attacker=None,
                            defender=defender, batch_size=self.batch_size, num_episodes=self.num_episodes)

        model = MNISTClassifier()
        simulator3 = Simulator(self.X_mnist_tr, self.y_mnist_tr, model, attacker=attacker,
                            defender=None,batch_size=self.batch_size, num_episodes=self.num_episodes)

        model = MNISTClassifier()
        simulator4 = Simulator(self.X_mnist_tr, self.y_mnist_tr, model, attacker=None,
                               defender=None, batch_size=self.batch_size, num_episodes=self.num_episodes)

        #simulate attack and defense separately using class method
        simulator1.run()
        simulator2.run()
        simulator3.run()
        simulator4.run()

        simulators = {'attacker_and_defense': simulator1, 'only_defender':simulator2,
                    'only_attacker': simulator3, 'regular': simulator4}

        wrapped_data, wrapped_models =  wrap_results(simulators)

    def test_iris(self):
        """Attack and defense combinations simulations for Iris classifier."""

        # Instantiate necessary classes
        # Instantiate necessary classes
        defender = FeasibleSetDefender(self.X_iris_tr, self.y_iris_tr, 0.5, one_hot=True)
                                #SoftmaxDefender(threshold=0.1))
        
        attacker = AddLabeledPointsAttacker(0.6, 1, one_hot=True)

        #implement attack and defense strategies through learner
        model = IrisClassifier()
        simulator1 = Simulator(self.X_iris_tr, self.y_iris_tr, model, attacker=attacker,
                            defender=defender, batch_size=self.batch_size, num_episodes=self.num_episodes)

        model = IrisClassifier()
        simulator2 = Simulator(self.X_iris_tr, self.y_iris_tr, model, attacker=None,
                            defender=defender, batch_size=self.batch_size, num_episodes=self.num_episodes)

        model = IrisClassifier()
        simulator3 = Simulator(self.X_iris_tr, self.y_iris_tr, model, attacker=attacker,
                            defender=None, batch_size=self.batch_size, num_episodes=self.num_episodes)

        model = IrisClassifier()
        simulator4 = Simulator(self.X_iris_tr, self.y_iris_tr, model, attacker=None,
                            defender=None, batch_size=self.batch_size, num_episodes=self.num_episodes)

        #simulate attack and defense separately using class method
        simulator1.run()
        simulator2.run()
        simulator3.run()
        simulator4.run()

        simulators = {'attacker_and_defense': simulator1, 'only_defender':simulator2,
                    'only_attacker': simulator3, 'regular': simulator4}

        wrapped_data, wrapped_models =  wrap_results(simulators)


# =============================================================================
#  MAIN ENTRY POINT
# =============================================================================

if __name__ == '__main__':
    unittest.main()