
# Written by: Oskar
# Last edited: 2022/01/22
# Description: Example main pipeline execution script. This is a rough example
# of how the pipeline could look, and how the various classes could interact
# with one another. We can write this script properly on Tuesday :)


# =============================================================================
#  IMPORTS AND DEPENDENCIES
# =============================================================================

import numpy as np

#pypoison modules
from data import DataLoader
from attack import AddLabeledPointsAttacker, RandomAttacker, LabelFlipperAttacker
from defence import FeasibleSetDefender, DefenderGroup, SoftmaxDefender
from model import IrisClassifier, MNISTClassifier
from postprocessing import PostProcessor
from simulation import Simulator, wrap_results
from utils import train_test_iris, train_test_MNIST
import torch

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
    #split iris dataset into train and test
    X_train, y_train, X_test, y_test = train_test_iris(num_stacks=10)

    # Instantiate necessary classes
    # Instantiate necessary classes
    defender = FeasibleSetDefender(X_train, y_train, 0.5, one_hot=True)
                             #SoftmaxDefender(threshold=0.1))
    
    # attacker = AddLabeledPointsAttacker(0.6, 1, one_hot=True)
    dict = {0:1}
    attacker = LabelFlipperAttacker(1, dict, True)

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

    wrapped_results_X, wrapped_results_y, wrapped_models =  wrap_results(simulators)

    postprocessor = PostProcessor(wrapped_models, BATCH_SIZE, NUM_EPISODES, model)
    postprocessor.plot_online_learning_accuracies(X_test, y_test, save=False)

def test_iris_regular():
    """No attack and defense trial on Iris dataset."""
    #split iris dataset into train and test
    BATCH_SIZE=10
    NUM_EPISODES=500
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

## ============================================================================
## Test MNIST Classifier
## ============================================================================
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
    simulator1.run(defender_requires_model=True)
    simulator2.run(defender_requires_model=True)
    simulator3.run(defender_requires_model=True)
    simulator4.run(defender_requires_model=True)

    simulators = {'attacker_and_defense': simulator1, 'only_defender':simulator2,
                'only_attacker': simulator3, 'regular': simulator4}

    wrapped_results_X, wrapped_results_y, wrapped_models =  wrap_results(simulators)

    postprocessor = PostProcessor(wrapped_models, BATCH_SIZE, NUM_EPISODES, model)
    postprocessor.plot_online_learning_accuracies(X_test, y_test, save=False)


## ============================================================================
## Test PostProcessor
## ============================================================================
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
        wrapped_results_X, wrapped_results_y, wrapped_models =  wrap_results(simulators)

        torch.save(wrapped_models, 'wrapped_models.pickle')
        torch.save(model, 'baseline.pickle')
    else:
        wrapped_models = torch.load(saved_models)
        baseline_model = torch.load(baseline)

    postprocessor = PostProcessor(wrapped_models, BATCH_SIZE, NUM_EPISODES, baseline_model)
    postprocessor.plot_decision_boundaries(X_test, y_test, num_points = 2000, perplexity=100, 
                                           n_iter=2000, fontsize=13, markersize=20, figsize=(10,10), 
                                           resolution=0.2, save=True)

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
        wrapped_results_X, wrapped_results_y, wrapped_models =  wrap_results(simulators)

        torch.save(wrapped_models, 'wrapped_models.pickle')
        torch.save(model, 'baseline.pickle')
    else:
        wrapped_models = torch.load(saved_models)
        baseline_model = torch.load(baseline)

    postprocessor = PostProcessor(wrapped_models, BATCH_SIZE, NUM_EPISODES, baseline_model)
    postprocessor.plot_decision_boundaries(X_test, y_test, num_points = 200, perplexity=100, 
                                           n_iter=2000, fontsize=13, markersize=20, figsize=(10,10), 
                                           resolution=0.2, save=True)

# =============================================================================
#  MAIN ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    #-----------IRIS TRIALS------------
    # test_iris_simulations()
    #test_iris_regular()

    #-----------MNIST TRIALS-----------
    #test_MNIST_regular()
    #test_MNIST_simulations()

    #----------POSTPROCESSOR TRIALS----
    saved_models='wrapped_models.pickle'
    baseline = 'baseline.pickle'
    test_decision_boundaries_MNIST(saved_models=None, baseline=None)
    #test_decision_boundaries_iris()


