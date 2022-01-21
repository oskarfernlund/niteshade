# Main pipeline execution script.

# =============================================================================
#  IMPORTS AND DEPENDENCIES
# =============================================================================

import numpy as np

from datastream import DataStream
from attack import RandomAttacker
from defence import RandomDefender
from model import Classifier
from postprocessing import PostProcessor


# =============================================================================
#  GLOBAL VARIABLES
# =============================================================================

# Dataset
DATAFILE = "iris.data"

# Data stream
BATCH_SIZE = 1

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
    dataset = np.load(DATAFILE)
    X, y = dataset[:, :-1], dataset[:, -1]

    # Instantiate necessary classes
    datastream = DataStream(X, y)
    attacker = RandomAttacker()
    defender = RandomDefender()
    model = Classifier(HIDDEN_NEURONS, OPTIMISER, LEARNING_RATE)
    postprocessor = PostProcessor()

    # Online learning loop
    while not datastream.is_over():

        # Fetch a new datapoint (or batch) from the stream
        datapoint = datastream.fetch()
        
        # Attacker's turn to perturb
        perturbed_datapoint = attacker.perturb(datapoint)

        # Defender's turn to defend
        if defender.rejects(perturbed_datapoint):
            continue
        else:
            model.optimiser.step(perturbed_datapoint)

        # Postprocessor saves results
        postprocessor.cache(datapoint, perturbed_datapoint, model.epoch_loss)

    # Save the results to the results directory
    postprocessor.save_results()

            
# =============================================================================
#  MAIN ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()
