
# Written by: Mart
# Last edited: 2022/01/23
# Description: Defender class, outlines various defender classes


# =============================================================================
#  IMPORTS AND DEPENDENCIES
# =============================================================================

import numpy as np

from datastream import DataStream
from model import Classifier


# =============================================================================
#  GLOBAL VARIABLES
# =============================================================================


# =============================================================================
#  RandomDefender class
# =============================================================================
class RandomDefender:
    # Simple RandomDefender who will reject a datapoint randomly depending on the input rate
    def __init__(self, rate) -> None:
        self.rate = rate
    
    def rejects(self,datapoint):
        #NB datapoint var actually not used but is declared as other defenders will use datapoint
        return np.random.rand() <= self.rate




# =============================================================================
#  FUNCTIONS
# =============================================================================

# =============================================================================
#  MAIN ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    pass
