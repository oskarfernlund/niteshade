#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Unit and integration tests for the simulation module.
"""


# =============================================================================
#  IMPORTS AND DEPENDENCIES
# =============================================================================
from matplotlib.pyplot import isinteractive
import pytest

from niteshade.attack import AddLabeledPointsAttacker, LabelFlipperAttacker, Attacker, AddPointsAttacker, PerturbPointsAttacker
from niteshade.defence import Defender, FeasibleSetDefender
from niteshade.models import IrisClassifier, MNISTClassifier
from niteshade.simulation import Simulator, wrap_results
from niteshade.utils import train_test_iris, train_test_MNIST

import torch.nn as nn
import torch
import numpy as np

# =============================================================================
#  Tests
# =============================================================================
@pytest.mark.long
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


@pytest.mark.long
def test_iris():
    """Attack and defense combinations simulations for Iris classifier."""
    batch_size = 5
    num_episodes = 10
    X_train, y_train, X_test, y_test = train_test_iris()
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

def test_attacker_arguments():
    class TestModelAttacker(Attacker):
        def __init__(self):
            super().__init__()
        def attack(self, X, y, model, arg1, arg2):
            assert isinstance(model, nn.Module)
            assert isinstance(X, (torch.Tensor, np.ndarray))
            assert isinstance(y, (torch.Tensor, np.ndarray))
            assert arg1 == 2
            assert arg2 == 5
            return X, y

    class TestAttacker(Attacker):
        def __init__(self):
            super().__init__()
        def attack(self, X, y, arg1, arg2):
            assert isinstance(X, (torch.Tensor, np.ndarray))
            assert isinstance(y, (torch.Tensor, np.ndarray))
            assert arg1 == 2
            assert arg2 == 5
            return X, y
    X_train, y_train, X_test, y_test = train_test_iris()
    batch_size = 5
    num_episodes = 10
    args = {"arg1": 2, "arg2": 5}

    model1 = IrisClassifier()
    attacker1 = TestModelAttacker()
    simulator1 = Simulator(X_train, y_train, model1, attacker=attacker1,
                           batch_size=batch_size, num_episodes=num_episodes)

    model2 = IrisClassifier()
    attacker2 = TestAttacker()
    simulator2 = Simulator(X_train, y_train, model2, attacker=attacker2,
                           batch_size=batch_size, num_episodes=num_episodes)

    simulator1.run(attacker_requires_model=True, attacker_args=args)
    simulator2.run(attacker_args=args)


# =============================================================================
#  MAIN ENTRY POINT
# =============================================================================

if __name__ == '__main__':
    test_attacker_arguments()
    test_mnist()
    test_iris()