#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Unit tests for out-of-the-box model classes.
"""


# =============================================================================
#  IMPORTS AND DEPENDENCIES
# =============================================================================

import pytest

from niteshade.attack import AddLabeledPointsAttacker, RandomAttacker, LabelFlipperAttacker
from niteshade.defence import FeasibleSetDefender, DefenderGroup, SoftmaxDefender

from niteshade.postprocessing import PostProcessor, PDF

from niteshade.models import IrisClassifier, MNISTClassifier, CifarClassifier
from niteshade.simulation import Simulator
from niteshade.utils import train_test_iris, train_test_MNIST, train_test_cifar


# =============================================================================
#  Tests
# =============================================================================
def test_point_counter_iris():
    batch_size = 1
    num_episodes = 100   
    X_train, y_train, X_test, y_test = train_test_iris()

    defender = FeasibleSetDefender(X_train, y_train, 0.5, one_hot=True)
    attacker = AddLabeledPointsAttacker(0.6, 1, one_hot=True)
    
    simulator1 = Simulator(X_train, y_train, IrisClassifier(), 
                           attacker=None, defender=None, 
                           batch_size=batch_size, num_episodes=num_episodes)
    simulator2 = Simulator(X_train, y_train, IrisClassifier(), 
                           attacker=attacker, defender=None, 
                           batch_size=batch_size, num_episodes=num_episodes)
    simulator3 = Simulator(X_train, y_train, IrisClassifier(), 
                           attacker=None, defender=defender, 
                           batch_size=batch_size, num_episodes=num_episodes)
    simulator4 = Simulator(X_train, y_train, IrisClassifier(), 
                           attacker=attacker, defender=defender, 
                           batch_size=batch_size, num_episodes=num_episodes)

    simulator1.run()
    simulator2.run()
    simulator3.run()
    simulator4.run()

    simulators = {'regular': simulator1, 
                  'only_attacker':simulator2,
                  'only_defender': simulator3, 
                  'attack_and_defence': simulator4}

    print(simulators['regular'])
    postprocessor = PostProcessor(simulators)

    data_modifications = postprocessor.track_data_modifications()

    # Sense check regular results
    regular_results = data_modifications['regular']
    assert regular_results['poisoned'] == 0
    assert regular_results['not_poisoned'] == regular_results['original_points_total']
    assert regular_results['correctly_defended'] == 0
    assert regular_results['incorrectly_defended'] == 0
    assert regular_results['training_points_total'] == regular_results['original_points_total']

    # Sense check attacker only results
    attacker_only_results = data_modifications['only_attacker']
    assert attacker_only_results['correctly_defended'] == 0
    assert attacker_only_results['incorrectly_defended'] == 0

    # Sense check defender only results
    defender_only_results = data_modifications['only_defender']
    assert defender_only_results['poisoned'] == 0
    assert defender_only_results['not_poisoned'] == defender_only_results['original_points_total']
    
    # Sense check attacker and defender results
    attack_and_defence_results = data_modifications['attack_and_defence']

    assert attack_and_defence_results['poisoned']+\
        attack_and_defence_results['not_poisoned']-\
        attack_and_defence_results['correctly_defended']-\
        attack_and_defence_results['incorrectly_defended'] ==\
        attack_and_defence_results['training_points_total']


def test_point_counter_MNIST():
    batch_size = 32
    num_episodes = 10   
    X_train, y_train, X_test, y_test = train_test_MNIST()

    defender = FeasibleSetDefender(X_train, y_train, 2000)
    attacker = AddLabeledPointsAttacker(0.6, 1)
    
    simulator1 = Simulator(X_train, y_train, MNISTClassifier(), 
                           attacker=None, defender=None, 
                           batch_size=batch_size, num_episodes=num_episodes)
    simulator2 = Simulator(X_train, y_train, MNISTClassifier(), 
                           attacker=attacker, defender=None, 
                           batch_size=batch_size, num_episodes=num_episodes)
    simulator3 = Simulator(X_train, y_train, MNISTClassifier(), 
                           attacker=None, defender=defender, 
                           batch_size=batch_size, num_episodes=num_episodes)
    simulator4 = Simulator(X_train, y_train, MNISTClassifier(), 
                           attacker=attacker, defender=defender, 
                           batch_size=batch_size, num_episodes=num_episodes)

    simulator1.run()
    simulator2.run()
    simulator3.run()
    simulator4.run()

    simulators = {'regular': simulator1, 
                  'only_attacker':simulator2,
                  'only_defender': simulator3, 
                  'attack_and_defence': simulator4}

    postprocessor = PostProcessor(simulators)

    data_modifications = postprocessor.track_data_modifications()

    # Sense check regular results
    regular_results = data_modifications['regular']
    assert regular_results['poisoned'] == 0
    assert regular_results['not_poisoned'] == regular_results['original_points_total']
    assert regular_results['correctly_defended'] == 0
    assert regular_results['incorrectly_defended'] == 0
    assert regular_results['training_points_total'] == regular_results['original_points_total']

    # Sense check attacker only results
    attacker_only_results = data_modifications['only_attacker']
    assert attacker_only_results['correctly_defended'] == 0
    assert attacker_only_results['incorrectly_defended'] == 0

    # Sense check defender only results
    defender_only_results = data_modifications['only_defender']
    assert defender_only_results['poisoned'] == 0
    assert defender_only_results['not_poisoned'] == defender_only_results['original_points_total']
    
    # Sense check attacker and defender results
    attack_and_defence_results = data_modifications['attack_and_defence']

    assert attack_and_defence_results['poisoned']+\
        attack_and_defence_results['not_poisoned']-\
        attack_and_defence_results['correctly_defended']-\
        attack_and_defence_results['incorrectly_defended'] ==\
        attack_and_defence_results['training_points_total']

# =============================================================================
#  MAIN ENTRY POINT
# =============================================================================

if __name__ == '__main__':
    test_point_counter_iris()
    test_point_counter_MNIST()