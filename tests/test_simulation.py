#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Unit and integration tests for the simulation module.
"""


# =============================================================================
#  IMPORTS AND DEPENDENCIES
# =============================================================================
import pytest

from niteshade.attack import AddLabeledPointsAttacker, LabelFlipperAttacker, Attacker
from niteshade.defence import Defender, FeasibleSetDefender
from niteshade.models import IrisClassifier, MNISTClassifier
from niteshade.simulation import Simulator, wrap_results
from niteshade.utils import train_test_iris, train_test_MNIST


# =============================================================================
#  Tests
# =============================================================================
def test_mnist():
    """Attack and defense combinations simulations for Iris classifier."""
    batch_size = 32
    num_episodes = 10
    X_train, y_train, X_test, y_test = train_test_MNIST()
    # Instantiate necessary classes
    # Instantiate necessary classes
    defender = FeasibleSetDefender(X_train, y_train, 2000)
    # defender = SoftmaxDefender(threshold=0.1)
    # attacker = SimpleAttacker(0.6, 1)
    
    dict = {1:4, 4:1, 3:5, 5:3}
    attacker = LabelFlipperAttacker(1, dict) 

    #implement attack and defense strategies through learner
    model = MNISTClassifier()
    simulator1 = Simulator(X_train, y_train, model, attacker=attacker,
                        defender=defender, batch_size=batch_size, num_episodes=num_episodes)

    model = MNISTClassifier()
    simulator2 = Simulator(X_train, y_train, model, attacker=None,
                        defender=defender, batch_size=batch_size, num_episodes=num_episodes)

    model = MNISTClassifier()
    simulator3 = Simulator(X_train, y_train, model, attacker=attacker,
                        defender=None, batch_size=batch_size, num_episodes=num_episodes)

    model = MNISTClassifier()
    simulator4 = Simulator(X_train, y_train, model, attacker=None,
                            defender=None, batch_size=batch_size, num_episodes=num_episodes)

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
    batch_size = 32
    num_episodes = 10
    X_train, y_train, X_test, y_test = train_test_iris(num_stacks=1)
    # Instantiate necessary classes
    # Instantiate necessary classes
    defender = FeasibleSetDefender(X_train, y_train, 0.5, one_hot=True)
                            #SoftmaxDefender(threshold=0.1))
    
    attacker = AddLabeledPointsAttacker(0.6, 1, one_hot=True)

    #implement attack and defense strategies through learner
    model = IrisClassifier()
    simulator1 = Simulator(X_train, y_train, model, attacker=attacker,
                        defender=defender, batch_size=batch_size, num_episodes=num_episodes)

    model = IrisClassifier()
    simulator2 = Simulator(X_train, y_train, model, attacker=None,
                        defender=defender, batch_size=batch_size, num_episodes=num_episodes)

    model = IrisClassifier()
    simulator3 = Simulator(X_train, y_train, model, attacker=attacker,
                        defender=None, batch_size=batch_size, num_episodes=num_episodes)

    model = IrisClassifier()
    simulator4 = Simulator(X_train, y_train, model, attacker=None,
                        defender=None, batch_size=batch_size, num_episodes=num_episodes)

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
    test_mnist()
    test_iris()