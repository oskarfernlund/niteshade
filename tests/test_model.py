#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Unit tests for out-of-the-box model classes.
"""


# =============================================================================
#  IMPORTS AND DEPENDENCIES
# =============================================================================

import pytest

from niteshade.models import IrisClassifier, MNISTClassifier, CifarClassifier
from niteshade.simulation import Simulator
from niteshade.utils import train_test_iris, train_test_MNIST, train_test_cifar


# =============================================================================
#  Tests
# =============================================================================
def test_iris():
    """No attack and defense trial on Iris dataset."""
    batch_size = 32
    num_episodes = 10   
    #split iris dataset into train and test
    X_train, y_train, X_test, y_test = train_test_iris(num_stacks=10)

    #implement attack and defense strategies through learner
    model = IrisClassifier()
    simulator = Simulator(X_train, y_train, model, attacker=None,
                        defender=None, batch_size=batch_size, num_episodes=num_episodes)

    #simulate attack and defense separately using run() method
    simulator.run()

    #evaluate on test set
    test_accuracy = simulator.model.evaluate(X_test, y_test, batch_size)  

def test_MNIST():
    batch_size = 32
    num_episodes = 10   
    X_train, y_train, X_test, y_test = train_test_MNIST()

    #implement attack and defense strategies through learner
    model = MNISTClassifier()
    simulator = Simulator(X_train, y_train, model, attacker=None, defender=None, 
                          batch_size=batch_size, num_episodes=num_episodes)

    #simulate attack and defense separately using run() method
    simulator.run()

    #evaluate on test set
    test_accuracy = simulator.model.evaluate(X_test, y_test, batch_size)  

def test_CIFAR():
    batch_size = 5
    num_episodes = 5
    X_train, y_train, X_test, y_test = train_test_cifar()

    #implement attack and defense strategies through learner
    model = CifarClassifier()
    simulator = Simulator(X_train, y_train, model, attacker=None, defender=None, 
                          batch_size=batch_size, num_episodes=num_episodes)

    for _ in range(2):
        #simulate attack and defense separately using run() method
        simulator.run()

    #evaluate on test set
    test_accuracy = model.evaluate(X_test, y_test, batch_size) 

# =============================================================================
#  MAIN ENTRY POINT
# =============================================================================

if __name__ == '__main__':
    test_iris()
    test_MNIST()
    test_CIFAR()