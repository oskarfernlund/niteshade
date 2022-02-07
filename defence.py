
# Written by: Mart
# Last edited: 2022/01/23
# Description: Defender class, outlines various defender classes


# =============================================================================
#  IMPORTS AND DEPENDENCIES
# =============================================================================

import numpy as np
from datastream import DataStream
from model import IrisClassifier


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
#  FeasibleSetDefender class
# =============================================================================

class FeasibleSetDefender:
    #Extremely simple class_mean_based outlier detector
    def __init__(self, initial_dataset_x, initial_dataset_y, threshold) -> None:
        self._init_x = initial_dataset_x
        self._init_y = initial_dataset_y
        self._feasible_set_construction()
        self._threshold = threshold

    def _feasible_set_construction(self):
        #Implement feasible set construction
        # Currently implemented for 1d datapoints, tabular data
        labels = np.unique(self._init_y)
        feasible_set = {}
        label_counts = {}
        for label in labels:
            label_rows = (self._init_y == label)
            label_counts[label] = np.sum(label_rows)
            feasible_set[label] = np.mean(self._init_x[label_rows], 0)
        self.feasible_set = feasible_set
        self._label_counts = label_counts

    def _feasible_set_adjustment(self, datapoint, label):
        #Adjust running means of feasible set
        label_mean = self.feasible_set[label]
        self._label_counts[label]+=1
        new_mean = label_mean + (datapoint-label_mean)/self._label_counts[label]
        self.feasible_set[label] = new_mean
    
    def _distance_metric(self,datapoint, label):
        #Calculate the distance metric for the datapoint from the feasible set mean
        label_mean = self.feasible_set[label]
        #simple eucl mean
        distance = np.sqrt(np.sum((datapoint - label_mean)**2))
        return distance

    def rejects(self,datapoint, label):
        #Reject datapoint taking into account running means
        data_label = label[0]
        distance = self._distance_metric(datapoint, data_label)
        if distance > self._threshold:
            return True
        else: 
            self._feasible_set_adjustment(datapoint, data_label)
            return False
        



# =============================================================================
#  FUNCTIONS
# =============================================================================

# =============================================================================
#  MAIN ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    x = np.array([[1,2,3], [1,3,2], [3,4,5]])
    y = np.array([1,2,1])
    defender = FeasibleSetDefender(x,y, 3)
    datapoint = np.array([2,2,2])
    label = np.array([1])
    print(defender.rejects(datapoint, label))


    
