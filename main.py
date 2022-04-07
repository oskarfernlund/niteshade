
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
from attack import SimpleAttacker, RandomAttacker, LabelFlipperAttacker
from defence import FeasibleSetDefender, DefenderGroup, SoftmaxDefender
from model import IrisClassifier, MNISTClassifier
from postprocessing import PostProcessor
from simulation import Simulator, wrap_results
from utils import train_test_iris, train_test_MNIST

import torchvision


# =============================================================================
#  GLOBAL VARIABLES
# =============================================================================
# batch size
BATCH_SIZE = 256
NUM_EPISODES = 30

# Model
# HIDDEN_NEURONS = (4, 16, 3) automicatically set in IrisClassifier
# ACTIVATIONS = ("relu", "softmax")  automicatically set in IrisClassifier
OPTIMISER = "adam"
LOSS_FUNC = "cross_entropy"
LEARNING_RATE = 0.01

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
    defender_kwargs = {"requires_model": True}
    
    attacker = SimpleAttacker(0.6, 1, one_hot=True)
    
    # label_flips_dict = {1:4, 4:1, 3:5, 5:3}
    # attacker = LabelFlipperAttacker(1, label_flips_dict)

    #implement attack and defense strategies through learner
    model = IrisClassifier(OPTIMISER, LOSS_FUNC, LEARNING_RATE)
    simulator1 = Simulator(X_train, y_train, model, attacker=attacker,
                        defender=defender, batch_size=BATCH_SIZE, num_episodes=NUM_EPISODES)

    model = IrisClassifier(OPTIMISER, LOSS_FUNC, LEARNING_RATE)
    simulator2 = Simulator(X_train, y_train, model, attacker=None,
                        defender=defender, batch_size=BATCH_SIZE, num_episodes=NUM_EPISODES)

    model = IrisClassifier(OPTIMISER, LOSS_FUNC, LEARNING_RATE)
    simulator3 = Simulator(X_train, y_train, model, attacker=attacker,
                        defender=None, batch_size=BATCH_SIZE, num_episodes=NUM_EPISODES)

    model = IrisClassifier(OPTIMISER, LOSS_FUNC, LEARNING_RATE)
    simulator4 = Simulator(X_train, y_train, model, attacker=None,
                        defender=None, batch_size=BATCH_SIZE, num_episodes=NUM_EPISODES)

    #simulate attack and defense separately using class method
    simulator1.run(defender_kwargs = defender_kwargs)
    simulator2.run(defender_kwargs = defender_kwargs)
    simulator3.run(defender_kwargs = defender_kwargs)
    simulator4.run(defender_kwargs = defender_kwargs)

    simulators = {'attacker_and_defense': simulator1, 'only_defender':simulator2,
                'only_attacker': simulator3, 'regular': simulator4}

    wrapped_results_X, wrapped_results_y, wrapped_models =  wrap_results(simulators)

    postprocessor = PostProcessor(wrapped_models, BATCH_SIZE, NUM_EPISODES, model)
    postprocessor.plot_online_learning_accuracies(X_test, y_test, save=False)

def test_iris_regular():
    """No attack and defense trial on Iris dataset."""
    #split iris dataset into train and test
    X_train, y_train, X_test, y_test = train_test_iris(num_stacks=10)

    #implement attack and defense strategies through learner
    model = IrisClassifier(OPTIMISER, LOSS_FUNC, LEARNING_RATE)
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
    X_train, y_train, X_test, y_test = train_test_MNIST()

    #implement attack and defense strategies through learner
    model = MNISTClassifier(OPTIMISER, LOSS_FUNC, LEARNING_RATE)
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
    # attacker = SimpleAttacker(0.6, 1)
    
    dict = {1:4, 4:1, 3:5, 5:3}
    attacker = LabelFlipperAttacker(1, dict) 

    #implement attack and defense strategies through learner
    model = MNISTClassifier(OPTIMISER, LOSS_FUNC, LEARNING_RATE)
    simulator1 = Simulator(X_train, y_train, model, attacker=attacker,
                        defender=defender, batch_size=BATCH_SIZE, num_episodes=NUM_EPISODES)

    model = MNISTClassifier(OPTIMISER, LOSS_FUNC, LEARNING_RATE)
    simulator2 = Simulator(X_train, y_train, model, attacker=None,
                        defender=defender, batch_size=BATCH_SIZE, num_episodes=NUM_EPISODES)

    model = MNISTClassifier(OPTIMISER, LOSS_FUNC, LEARNING_RATE)
    simulator3 = Simulator(X_train, y_train, model, attacker=attacker,
                        defender=None, batch_size=BATCH_SIZE, num_episodes=NUM_EPISODES)

    model = MNISTClassifier(OPTIMISER, LOSS_FUNC, LEARNING_RATE)
    simulator4 = Simulator(X_train, y_train, model, attacker=None,
                        defender=None, batch_size=BATCH_SIZE, num_episodes=NUM_EPISODES)

    #simulate attack and defense separately using class method
    simulator1.run(defender_requires_model=True)
    print("yay")
    simulator2.run(defender_requires_model=True)
    print("yayyy")
    simulator3.run(defender_requires_model=True)
    simulator4.run(defender_requires_model=True)

    simulators = {'attacker_and_defense': simulator1, 'only_defender':simulator2,
                'only_attacker': simulator3, 'regular': simulator4}

    wrapped_results_X, wrapped_results_y, wrapped_models =  wrap_results(simulators)

    postprocessor = PostProcessor(wrapped_models, BATCH_SIZE, NUM_EPISODES, model)
    postprocessor.plot_online_learning_accuracies(X_test, y_test, save=False)

# =============================================================================
#  MAIN ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    #-----------IRIS TRIALS------------
    #test_iris_simulations()
    #test_iris_regular()

    #-----------MNIST TRIALS-----------
    #test_MNIST_regular()
    test_MNIST_simulations()

