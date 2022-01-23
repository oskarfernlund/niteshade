
# Written by: Oskar
# Last edited: 2022/01/22
# Description: Example main pipeline execution script. This is a rough example
# of how the pipeline could look, and how the various classes could interact
# with one another. We can write this script properly on Tuesday :)


# =============================================================================
#  IMPORTS AND DEPENDENCIES
# =============================================================================

import numpy as np

from datastream import DataStream
from attack import RandomAttacker
from defence import RandomDefender
from model import Classifier
from postprocessing import PostProcessor
from sklearn import datasets
from sklearn.utils import shuffle


# =============================================================================
#  GLOBAL VARIABLES
# =============================================================================

# Dataset
datafile = datasets.load_iris()

# Data stream
BATCH_SIZE = 10

# Model
HIDDEN_NEURONS = (16, 16)
OPTIMISER = "SGD"
LEARNING_RATE = 0.001


# =============================================================================
#  FUNCTIONS
# =============================================================================

def main():
    """ Main pipeline execution. """

    # Load dataset
    X = np.array(datafile.data)
    y = np.array(datafile.target)
    X, y = shuffle(X, y)

    # Instantiate necessary classes
    datastream = DataStream(X, y, BATCH_SIZE)
    defender = defender_initiator(defender_type = "RandomDefender", reject_rate = 0.1)
    model = Classifier(HIDDEN_NEURONS, OPTIMISER, LEARNING_RATE)
    postprocessor = PostProcessor()

    # Online learning loop
    while datastream.is_online():

        # Fetch a new datapoint (or batch) from the stream
        databatch = datastream.fetch()
        
        # Attacker's turn to perturb
        attacker = RandomAttacker(databatch)
        perturbed_databatch = attacker.perturb(databatch)

        # Defender's turn to defend
        if defender.rejects(perturbed_databatch):
            continue
        else:
            model.optimiser.step(perturbed_databatch)

        # Postprocessor saves results
        postprocessor.cache(databatch, perturbed_databatch, model.epoch_loss)

    # Save the results to the results directory
    postprocessor.save_results()

def defender_initiator(**kwargs):
    # Returns a defender class depending on which strategy we are using
    # Currently only the RandomDefender is implemented, for who a reject_rate arg needs to be passed in
    for key, value in kwargs.items():
        if key == "defender_type":
            if value =="RandomDefender":
                rate = kwargs["reject_rate"]
                return RandomDefender(rate)
            
# =============================================================================
#  MAIN ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()
