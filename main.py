
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
from attack import SimpleAttacker, RandomAttacker
from defence import FeasibleSetDefender, DefenderGroup, SoftmaxDefender
from model import IrisClassifier, MNISTClassifier
from postprocessing import PostProcessor
from simulation import Simulator, wrap_results

#sklearn & torch utils for testing
from sklearn.datasets import load_iris
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

import torchvision


# =============================================================================
#  GLOBAL VARIABLES
# =============================================================================
# batch size
BATCH_SIZE = 64
EPISODE_SIZE = 5

# Model
# HIDDEN_NEURONS = (4, 16, 3) automicatically set in IrisClassifier
# ACTIVATIONS = ("relu", "softmax")  automicatically set in IrisClassifier
OPTIMISER = "adam"
LOSS_FUNC = "cross_entropy"
LEARNING_RATE = 0.01

# =============================================================================
#  FUNCTIONS
# =============================================================================
def train_test_iris(num_stacks = 10):
    #define input and target data
    data = load_iris()

    #define input and target data
    X, y = data.data, data.target

    #one-hot encode
    enc = OneHotEncoder()
    y = enc.fit_transform(y.reshape(-1,1)).toarray()

    #stack data
    X = np.repeat(X, num_stacks, axis=0)
    y = np.repeat(y, num_stacks, axis=0)

    #split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    #normalise data using sklearn module
    scaler = MinMaxScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    X_train, y_train = shuffle(X_train, y_train)

    return X_train, y_train, X_test, y_test


def test_iris_simulations():
    """Attack and defense combinations simulations."""
    #split iris dataset into train and test
    X_train, y_train, X_test, y_test = train_test_iris(num_stacks=10)

    # Instantiate necessary classes
    # Instantiate necessary classes
    defender = DefenderGroup(FeasibleSetDefender(X_train, y_train, 0.5, one_hot=True),
                             SoftmaxDefender(threshold=0.1))
    defender_kwargs = {"requires_model": True}
    attacker = SimpleAttacker(0.6, 1, one_hot=True)

    #implement attack and defense strategies through learner
    model = IrisClassifier(OPTIMISER, LOSS_FUNC, LEARNING_RATE)
    simulator1 = Simulator(X_train, y_train, model, attacker=attacker,
                        defender=defender, batch_size=BATCH_SIZE, episode_size=EPISODE_SIZE)

    model = IrisClassifier(OPTIMISER, LOSS_FUNC, LEARNING_RATE)
    simulator2 = Simulator(X_train, y_train, model, attacker=None,
                        defender=defender, batch_size=BATCH_SIZE, episode_size=EPISODE_SIZE)

    model = IrisClassifier(OPTIMISER, LOSS_FUNC, LEARNING_RATE)
    simulator3 = Simulator(X_train, y_train, model, attacker=attacker,
                        defender=None, batch_size=BATCH_SIZE, episode_size=EPISODE_SIZE)

    model = IrisClassifier(OPTIMISER, LOSS_FUNC, LEARNING_RATE)
    simulator4 = Simulator(X_train, y_train, model, attacker=None,
                        defender=None, batch_size=BATCH_SIZE, episode_size=EPISODE_SIZE)

    #simulate attack and defense separately using class method
    simulator1.run(defender_kwargs = defender_kwargs)
    simulator2.run(defender_kwargs = defender_kwargs)
    simulator3.run(defender_kwargs = defender_kwargs)
    simulator4.run(defender_kwargs = defender_kwargs)

    simulators = {'attacker_and_defense': simulator1, 'only_defender':simulator2,
                'only_attacker': simulator3, 'regular': simulator4}

    wrapped_results_X, wrapped_results_y, wrapped_models =  wrap_results(simulators)

    postprocessor = PostProcessor(wrapped_models, BATCH_SIZE, EPISODE_SIZE, model)
    postprocessor.plot_online_learning_accuracies(X_test, y_test, save=False)

def test_iris_regular():
    """No attack and defense trial on Iris dataset."""
    #split iris dataset into train and test
    X_train, y_train, X_test, y_test = train_test_iris(num_stacks=10)

    #implement attack and defense strategies through learner
    model = IrisClassifier(OPTIMISER, LOSS_FUNC, LEARNING_RATE)
    simulator = Simulator(X_train, y_train, model, attacker=None,
                        defender=None, batch_size=BATCH_SIZE, episode_size=EPISODE_SIZE)

    #simulate attack and defense separately using run() method
    simulator.run()

    #evaluate on test set
    test_loss, test_accuracy = simulator.model.evaluate(X_test, y_test, BATCH_SIZE)  
    print(f"TEST LOSS; {test_loss}, TEST ACCURACY; {test_accuracy}")

## ============================================================================
## Test MNIST Classifier
## ============================================================================
def train_test_MNIST():
    MNIST_train = torchvision.datasets.MNIST('data/', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ]))

    #get inputs and labels and convert to numpy arrays
    X_train = MNIST_train.data.numpy().reshape(-1, 1, 28, 28)
    y_train = MNIST_train.targets.numpy()

    MNIST_test = torchvision.datasets.MNIST('data/', train=False, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ]))
    
    X_test = MNIST_test.data.numpy().reshape(-1, 1, 28, 28)
    y_test = MNIST_test.targets.numpy()

    return X_train, y_train, X_test, y_test

def test_MNIST_regular():
    X_train, y_train, X_test, y_test = train_test_MNIST()
    print(X_train.shape)
    print(y_train.shape)

    #implement attack and defense strategies through learner
    model = MNISTClassifier(OPTIMISER, LOSS_FUNC, LEARNING_RATE)
    simulator = Simulator(X_train, y_train, model, attacker=None,
                        defender=None, batch_size=BATCH_SIZE, episode_size=EPISODE_SIZE)

    #simulate attack and defense separately using run() method
    simulator.run()

    #evaluate on test set
    test_loss, test_accuracy = simulator.model.evaluate(X_test, y_test, BATCH_SIZE)  
    #print(f"TEST LOSS; {test_loss}, TEST ACCURACY; {test_accuracy}")

# =============================================================================
#  MAIN ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    #-----------IRIS TRIALS------------
    #test_iris_simulations()
    #test_iris_regular()

    #-----------MNIST TRIALS-----------
    test_MNIST_regular()

