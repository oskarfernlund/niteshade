#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Example pipeline execution script.
"""


# =============================================================================
#  IMPORTS AND DEPENDENCIES
# =============================================================================

import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from niteshade.attack import AddLabeledPointsAttacker, RandomAttacker, LabelFlipperAttacker
from niteshade.defence import FeasibleSetDefender, DefenderGroup, SoftmaxDefender
from niteshade.models import IrisClassifier, MNISTClassifier
from niteshade.postprocessing import PostProcessor, PDF
from niteshade.simulation import Simulator, wrap_results
from niteshade.utils import train_test_iris, train_test_MNIST, get_time_stamp_as_string, save_plot


# =============================================================================
#  GLOBAL VARIABLES
# =============================================================================

# batch size
BATCH_SIZE = 256
NUM_EPISODES = 20

# Model
# HIDDEN_NEURONS = (4, 16, 3) automicatically set in IrisClassifier
# ACTIVATIONS = ("relu", "softmax")  automicatically set in IrisClassifier


# =============================================================================
#  FUNCTIONS
# =============================================================================

def test_iris_simulations():
    """Attack and defense combinations simulations for Iris classifier."""
    BATCH_SIZE = 2
    NUM_EPISODES = 100
    #split iris dataset into train and test
    X_train, y_train, X_test, y_test = train_test_iris(num_stacks=1)

    # Instantiate necessary classes
    # Instantiate necessary classes
    defender = FeasibleSetDefender(X_train, y_train, 0.5, one_hot=True)
                             #SoftmaxDefender(threshold=0.1))
    
    attacker = AddLabeledPointsAttacker(0.6, 1, one_hot=True)

    #implement attack and defense strategies through learner
    model = IrisClassifier()
    simulator1 = Simulator(X_train, y_train, model, attacker=attacker,
                        defender=defender, batch_size=BATCH_SIZE, num_episodes=NUM_EPISODES)

    model = IrisClassifier()
    simulator2 = Simulator(X_train, y_train, model, attacker=None,
                        defender=defender, batch_size=BATCH_SIZE, num_episodes=NUM_EPISODES)

    model = IrisClassifier()
    simulator3 = Simulator(X_train, y_train, model, attacker=attacker,
                        defender=None, batch_size=BATCH_SIZE, num_episodes=NUM_EPISODES)

    model = IrisClassifier()
    simulator4 = Simulator(X_train, y_train, model, attacker=None,
                        defender=None, batch_size=BATCH_SIZE, num_episodes=NUM_EPISODES)

    #simulate attack and defense separately using class method
    simulator1.run()
    simulator2.run()
    simulator3.run()
    simulator4.run()

    simulators = {'attacker_and_defense': simulator1, 'only_defender':simulator2,
                'only_attacker': simulator3, 'regular': simulator4}

    wrapped_data, wrapped_models =  wrap_results(simulators)

    postprocessor = PostProcessor(wrapped_data, wrapped_models, BATCH_SIZE, NUM_EPISODES, model)


    # Get point counts for each simulation
    data_modifications = postprocessor.track_data_modifications()

    # Save the accruacy plot
    metrics = postprocessor.compute_online_learning_metrics(X_test, y_test)
    postprocessor.plot_online_learning_metrics(metrics, show_plot=False, save=True, 
                                                  plotname='test_accuracies', set_plot_title=False)
    
    # Generate a PDF
    time_stamp = get_time_stamp_as_string()
    header_title = f'Example Simulation Report as of {time_stamp}'
    pdf = PDF()
    pdf.set_title(header_title)
    pdf.add_table(data_modifications, 'Point Summary')
    pdf.add_all_charts_from_directory('output')
    pdf.output('summary_report.pdf', 'F')


def test_iris_regular():
    """No attack and defense trial on Iris dataset."""
    #split iris dataset into train and test
    BATCH_SIZE = 32
    NUM_EPISODES = 500
    X_train, y_train, X_test, y_test = train_test_iris(num_stacks=10)

    #implement attack and defense strategies through learner
    model = IrisClassifier()
    simulator = Simulator(X_train, y_train, model, attacker=None,
                          defender=None, batch_size=BATCH_SIZE, num_episodes=NUM_EPISODES)

    #simulate attack and defense separately using run() method
    simulator.run()

    #evaluate on test set
    test_loss, test_accuracy = simulator.model.evaluate(X_test, y_test, BATCH_SIZE)  
    print(f"TEST LOSS; {test_loss}, TEST ACCURACY; {test_accuracy}")


def test_MNIST_regular():
    BATCH_SIZE=64
    NUM_EPISODES=20
    X_train, y_train, X_test, y_test = train_test_MNIST()

    #implement attack and defense strategies through learner
    model = MNISTClassifier()
    simulator = Simulator(X_train, y_train, model, attacker=None, defender=None, 
                          batch_size=BATCH_SIZE, num_episodes=NUM_EPISODES)

    #simulate attack and defense separately using run() method
    simulator.run()

    #evaluate on test set
    test_loss, test_accuracy = simulator.model.evaluate(X_test, y_test, BATCH_SIZE)  
    #print(f"TEST LOSS; {test_loss}, TEST ACCURACY; {test_accuracy}")


def test_MNIST_simulations():
    """Attack and defense combinations simulations for Iris classifier."""
    #split iris dataset into train and test
    X_train, y_train, X_test, y_test = train_test_MNIST()

    # Instantiate necessary classes
    # Instantiate necessary classes
    defender = FeasibleSetDefender(X_train, y_train, 2000)
    # defender = SoftmaxDefender(threshold=0.1)
    
    attacker = AddLabeledPointsAttacker(0.6, 1)
    
    # dict = {1:4, 4:1, 3:5, 5:3}
    # attacker = LabelFlipperAttacker(1, dict) 

    #implement attack and defense strategies through learner
    model = MNISTClassifier()
    simulator1 = Simulator(X_train, y_train, model, attacker=attacker,
                        defender=defender, batch_size=BATCH_SIZE, num_episodes=NUM_EPISODES)

    model = MNISTClassifier()
    simulator2 = Simulator(X_train, y_train, model, attacker=None,
                        defender=defender, batch_size=BATCH_SIZE, num_episodes=NUM_EPISODES)

    model = MNISTClassifier()
    simulator3 = Simulator(X_train, y_train, model, attacker=attacker,
                        defender=None, batch_size=BATCH_SIZE, num_episodes=NUM_EPISODES)

    model = MNISTClassifier()
    simulator4 = Simulator(X_train, y_train, model, attacker=None,
                        defender=None, batch_size=BATCH_SIZE, num_episodes=NUM_EPISODES)

    #simulate attack and defense separately using class method
    simulator1.run()
    #simulator2.run()
    #simulator3.run()
    #simulator4.run()

    simulators = {'attacker_and_defense': simulator1}#, 'only_defender':simulator2,
                #'only_attacker': simulator3, 'regular': simulator4}

    wrapped_data, wrapped_models =  wrap_results(simulators)


def test_decision_boundaries_MNIST(saved_models=None, baseline=None):
    # batch size
    BATCH_SIZE = 128
    NUM_EPISODES = 30
    #split iris dataset into train and test
    X_train, y_train, X_test, y_test = train_test_MNIST()

    defender = FeasibleSetDefender(X_train, y_train, 2000)
    label_flips_dict = {1:9, 9:1}
    attacker = LabelFlipperAttacker(1, label_flips_dict)
    #attacker = SimpleAttacker(0.6, 1)

    model = MNISTClassifier()
    simulator1 = Simulator(X_train, y_train, model, attacker=attacker,
                        defender=None, batch_size=BATCH_SIZE, num_episodes=NUM_EPISODES)

    baseline_model = MNISTClassifier()
    simulator2 = Simulator(X_train, y_train, baseline_model, attacker=None,
                        defender=None, batch_size=BATCH_SIZE, num_episodes=NUM_EPISODES)
    
    model = MNISTClassifier()
    simulator3 = Simulator(X_train, y_train, model, attacker=None,
                        defender=defender, batch_size=BATCH_SIZE, num_episodes=NUM_EPISODES)

    model = MNISTClassifier()
    simulator4 = Simulator(X_train, y_train, model, attacker=attacker,
                        defender=defender, batch_size=BATCH_SIZE, num_episodes=NUM_EPISODES)
    

    #simulate attack and defense separately using class method
    if saved_models is None and baseline is None:
        simulator1.run()
        simulator2.run()
        #simulator3.run()
        simulator4.run()    

        simulators = {'only_attacker': simulator1, 'regular': simulator2,
                      'attacker_and_defender': simulator4}
        wrapped_data, wrapped_models =  wrap_results(simulators)

        torch.save(wrapped_models, 'wrapped_models.pickle')
        torch.save(model, 'baseline.pickle')
    else:
        wrapped_models = torch.load(saved_models)
        baseline_model = torch.load(baseline)

    postprocessor = PostProcessor(wrapped_data, wrapped_models, BATCH_SIZE, NUM_EPISODES, baseline_model)
    postprocessor.plot_decision_boundaries(X_test, y_test, num_points = 2000, perplexity=100, 
                                           n_iter=2000, fontsize=13, markersize=20, figsize=(10,10), 
                                           resolution=0.2, save=True, show_plot=False)

    # Get point counts for each simulation
    data_modifications = postprocessor.track_data_modifications()

    # Save the accruacy plot
    metrics = postprocessor.compute_online_learning_metrics(X_test, y_test)
    postprocessor.plot_online_learning_metrics(metrics, show_plot=False, save=True, 
                                                  plotname='test_accuracies', set_plot_title=False)
    
    time_stamp = get_time_stamp_as_string()
    header_title = f'Example Simulation Report as of {time_stamp}'
    pdf = PDF()
    pdf.set_title(header_title)
    pdf.add_table(data_modifications, 'Point Summary')
    pdf.add_all_charts_from_directory('output')
    pdf.output('summary_report.pdf', 'F')


def test_decision_boundaries_iris(saved_models=None, baseline=None):
    # batch size
    BATCH_SIZE = 128
    NUM_EPISODES = 30
    #split iris dataset into train and test
    X_train, y_train, X_test, y_test = train_test_iris(num_stacks=10)

    # Instantiate necessary classes
    # Instantiate necessary classes
    defender = FeasibleSetDefender(X_train, y_train, 0.5, one_hot=True)
                             #SoftmaxDefender(threshold=0.1))
    
    label_flips_dict = {0:2, 2:0}
    attacker = LabelFlipperAttacker(1, label_flips_dict, one_hot=True)
    
    # label_flips_dict = {1:4, 4:1, 3:5, 5:3}
    # attacker = LabelFlipperAttacker(1, label_flips_dict)

    #implement attack and defense strategies through learner
    model = IrisClassifier()
    simulator1 = Simulator(X_train, y_train, model, attacker=attacker,
                        defender=defender, batch_size=BATCH_SIZE, num_episodes=NUM_EPISODES)

    model = IrisClassifier()
    simulator2 = Simulator(X_train, y_train, model, attacker=None,
                        defender=defender, batch_size=BATCH_SIZE, num_episodes=NUM_EPISODES)

    model = IrisClassifier()
    simulator3 = Simulator(X_train, y_train, model, attacker=attacker,
                        defender=None, batch_size=BATCH_SIZE, num_episodes=NUM_EPISODES)

    baseline_model = IrisClassifier()
    simulator4 = Simulator(X_train, y_train, model, attacker=None,
                        defender=None, batch_size=BATCH_SIZE, num_episodes=NUM_EPISODES)

    #simulate attack and defense separately using class method
    if saved_models is None and baseline is None:
        simulator1.run()
        #simulator2.run()
        simulator3.run()
        simulator4.run()    

        simulators = {'only_attacker': simulator3, 'attacker_and_defender': simulator1,
                      'regular': simulator4}

        wrapped_data, wrapped_models =  wrap_results(simulators)

        torch.save(wrapped_models, 'wrapped_models.pickle')
        torch.save(model, 'baseline.pickle')
    else:
        wrapped_models = torch.load(saved_models)
        baseline_model = torch.load(baseline)

    postprocessor = PostProcessor(wrapped_data, wrapped_models, BATCH_SIZE, NUM_EPISODES, baseline_model)
    postprocessor.plot_decision_boundaries(X_test, y_test, num_points = 200, perplexity=100, 
                                           n_iter=2000, fontsize=13, markersize=20, figsize=(10,10), 
                                           resolution=0.2, save=True)


def test_learning_rates(saved_models=None, baseline=None):
    BATCH_SIZE = 128
    NUM_EPISODES = 30
    lrs = [0, 0.001, 0.003, 0.005, 0.008, 0.01, 0.03, 0.05, 0.08, 0.1, 0.5, 1]
    X_train, y_train, X_test, y_test = train_test_MNIST()

    defender = FeasibleSetDefender(X_train, y_train, 2000)
    
    label_flips_dict = {1:9, 9:1}
    attacker = LabelFlipperAttacker(1, label_flips_dict)


    baseline_model = MNISTClassifier(lr=0.01)
    simulator_regular = Simulator(X_train, y_train, baseline_model, attacker=attacker,
                        defender=None, batch_size=BATCH_SIZE, num_episodes=NUM_EPISODES)

    simulator_regular.run()
    simulators = {'regular_0.01': simulator_regular }

    for lr in lrs:
        model = MNISTClassifier(lr=lr)
        simulator = Simulator(X_train, y_train, model, attacker=attacker,
                    defender=defender, batch_size=BATCH_SIZE, num_episodes=NUM_EPISODES)
        simulator.run()
        simulators[f'lr_{lr}'] = simulator

    wrapped_data, wrapped_models =  wrap_results(simulators)

    torch.save(wrapped_models, 'wrapped_models.pickle')
    torch.save(model, 'baseline.pickle')

    postprocessor = PostProcessor(wrapped_data, wrapped_models, BATCH_SIZE, NUM_EPISODES, baseline_model)
    
    metrics = postprocessor.evaluate_simulators_metrics(X_test, y_test)
    
    fig, ax = plt.subplots()
    for key, value in metrics.items():
        x = float(key.split('_')[-1])
        y = value
        ax.scatter(x, y, label=key, marker='d', alpha=0.7)
        
    ax.set_xlabel('Learning Rate')
    ax.set_ylabel('Accuracy')
    plt.legend()
    plt.show()
    save_plot(fig)

# =============================================================================
#  MAIN ENTRY POINT
# =============================================================================


if __name__ == "__main__":
    #-----------IRIS TRIALS------------
    #test_iris_simulations()
    #test_iris_regular()

    #-----------MNIST TRIALS-----------
    #test_MNIST_regular()
    #test_MNIST_simulations()

    #----------POSTPROCESSOR TRIALS----
    #saved_models='wrapped_models.pickle'
    #baseline = 'baseline.pickle'
    #test_decision_boundaries_MNIST(saved_models=None, baseline=None)
    test_learning_rates(saved_models=None, baseline=None)
    #test_decision_boundaries_iris()


