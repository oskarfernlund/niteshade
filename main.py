
# Written by: Oskar
# Last edited: 2022/01/22
# Description: Example main pipeline execution script. This is a rough example
# of how the pipeline could look, and how the various classes could interact
# with one another. We can write this script properly on Tuesday :)


# =============================================================================
#  IMPORTS AND DEPENDENCIES
# =============================================================================

import numpy as np

from data import DataLoader
from attack import SimpleAttacker, RandomAttacker
from defence import RandomDefender, FeasibleSetDefender
from model import IrisClassifier
from postprocessing import PostProcessor
from simulation import Simulator, wrap_results


from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


# =============================================================================
#  GLOBAL VARIABLES
# =============================================================================
# Load dataset
data = np.loadtxt("datasets/iris.dat") #already contains one-hot encoding for targets

# batch size
BATCH_SIZE = 1
EPISODE_SIZE = 1

# Model
# HIDDEN_NEURONS = (4, 16, 3) automicatically set in IrisClassifier
# ACTIVATIONS = ("relu", "softmax")  automicatically set in IrisClassifier
OPTIMISER = "adam"
LOSS_FUNC = "cross_entropy"
LEARNING_RATE = 0.01
EPOCHS = 100


# =============================================================================
#  FUNCTIONS
# =============================================================================

def main():
    """ Main pipeline execution. (Trial with Iris dataset) """
    
    #define input and target data
    X, y = data[:, :4], data[:, 4:]

    #split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    #normalise data using sklearn module
    scaler = MinMaxScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    X_train, y_train = shuffle(X_train, y_train)

    # Instantiate necessary classes
    defender = FeasibleSetDefender(X_train, y_train, 0.5, one_hot=True)
    # defender = RandomDefender(0.3)
    attacker = SimpleAttacker(0.6, 1, one_hot=True)
    # attacker = RandomAttacker()
    model = IrisClassifier(OPTIMISER, LOSS_FUNC, LEARNING_RATE)

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
    simulator1.learn_online()
    simulator2.learn_online()
    simulator3.learn_online()
    simulator4.learn_online()

    simulators = {'regular': simulator1, 'only_defender':simulator2,
                'only_attacker': simulator3, 'attacker_and_defender': simulator4}

    wrapped_results_X, wrapped_results_y, wrapped_models =  wrap_results(simulators)

    #print("wrapped_results_X ", wrapped_results_X)
    #print("wrapped_results_y ", wrapped_results_y)
    #print("wrapped_models ", wrapped_models)

    test_loss, test_accuracy = simulator1.model.test(X_test, y_test, BATCH_SIZE)  

    postprocessor = PostProcessor(wrapped_models, BATCH_SIZE, EPISODE_SIZE, model)
    postprocessor.plot_online_learning_accuracies( X_test, y_test, save=False)
    #all_results_X, all_results_y, all_models = run_simulations(X_train, y_train, model, attacker=attacker,
    #                                                            defender=defender, batch_size=BATCH_SIZE, 
    #        

def defender_initiator(**kwargs):
    # Returns a defender class depending on which strategy we are using
    # Currently only the RandomDefender is implemented, for who a reject_rate arg needs to be passed in
    for key, value in kwargs.items():
        if key == "defender_type":
            if value =="RandomDefender":
                rate = kwargs["reject_rate"]
                return RandomDefender(rate)
            elif value =="FeasibleSetDefender":
                rate = kwargs["reject_rate"]
                return FeasibleSetDefender(rate)


# =============================================================================
#  MAIN ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    main()


