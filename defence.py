
# Written by: Mart
# Last edited: 2022/01/23
# Description: Defender class, outlines various defender classes


# =============================================================================
#  IMPORTS AND DEPENDENCIES
# =============================================================================

from re import T
import numpy as np
from data import DataLoader
from model import IrisClassifier
from abc import ABC, abstractmethod
import inspect
import torch



# =============================================================================
#  GLOBAL VARIABLES
# =============================================================================


# =============================================================================
#  GeneralDefender class
# =============================================================================
class DefenderGroup():
    def __init__(self, *defenders) -> None:
        self.defender_list = []
        for defender in defenders:
            self.defender_list.append(defender)
        
    def defend(self, X, y, **input_kwargs):
        for defender in self.defender_list:
            if len(X)>0:
                X, y = defender.defend(X, y, **input_kwargs)
        return X, y
        
# =============================================================================
#  GeneralDefender class
# =============================================================================
class Defender(ABC):
    def __init__(self) -> None:
        pass
    
    @abstractmethod
    def defend(self):
        raise NotImplementedError("Defend method needs to be implemented for a defender")


# =============================================================================
#  OutlierDefender class
# =============================================================================
class OutlierDefender(Defender):
    def __init__(self, initial_dataset_x, initial_dataset_y) -> None:
        super().__init__()
        self._init_x = initial_dataset_x
        self._init_y = initial_dataset_y

# =============================================================================
#  Softmax Defender class
# =============================================================================

class SoftmaxDefender(Defender):
    def __init__(self, threshold) -> None:
        super().__init__()
        self.threshold = threshold
    def defend(self, datapoints, labels, model, **input_kwargs):
        #convert np.ndarray to tensor for the NN
        X_batch = torch.tensor(datapoints)
        # Assume onehot for labels currently!!
        class_labels = torch.tensor(np.argmax(labels, axis = 1).reshape(-1,1))
        #zero gradients so they are not accumulated across batches
        model.optimizer.zero_grad()
        # Performs forward pass through classifier
        outputs = model.forward(X_batch.float())
        confidence = torch.gather(outputs, 1 , class_labels)
        mask = (confidence>self.threshold).squeeze(1)
        X_output = X_batch[mask].detach().numpy()
        y_output = labels[mask.numpy()]
        return (X_output, y_output)
        

        



# =============================================================================
#  FeasibleSetDefender class
# =============================================================================

class FeasibleSetDefender(OutlierDefender):
    #Extremely simple class_mean_based outlier detector
    def __init__(self, initial_dataset_x, initial_dataset_y, threshold, one_hot = False, dist_metric_type = "Eucleidian") -> None:
        super().__init__(initial_dataset_x, initial_dataset_y)
        self.one_hot = one_hot
        if self.one_hot:
            self._label_encoding()        
        self._feasible_set_construction()
        self._threshold = threshold
        self.distance_metric = Distance_metric(dist_metric_type)
    
    @property
    def distance_metric(self):
        return self.__distance_metric._type

    @distance_metric.setter
    def distance_metric(self, new_distance_metric):
        self.__distance_metric = new_distance_metric


    def _label_encoding(self):
        self._init_y=np.argmax(self._init_y, axis = 1)

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
    
    def _distance_metric_calculator(self,datapoint, label):
        #Calculate the distance metric for the datapoint from the feasible set mean
        label_mean = self.feasible_set[label]
        #simple eucl mean
        distance = self.__distance_metric.distance(datapoint, label_mean)
        return distance
    
    def defend(self,datapoints, labels, **input_kwargs):
        #Reject datapoint taking into account running means
        if self.one_hot:
            one_hot_length = len(labels[0])
            labels = np.argmax(labels, axis = 1)
        cleared_datapoints = []
        cleared_labels = []
        for id, datapoint in enumerate(datapoints):
            data_label = labels[id]
            distance = self._distance_metric_calculator(datapoint, data_label)
            if distance < self._threshold:
                self._feasible_set_adjustment(datapoint, data_label)
                cleared_datapoints.append(datapoint)
                cleared_labels.append(data_label)
        if len(cleared_labels) == 0:
            return (np.array([]), np.array([]))
        cleared_labels_stack = np.stack(cleared_labels)

        if self.one_hot:
            output_labels = np.zeros((len(cleared_labels_stack), one_hot_length))
            for id,label in enumerate(cleared_labels_stack):
                output_labels[id][label] = 1
            cleared_labels_stack = output_labels
        # Returns a tuple of np array of cleared datapoints and np array of cleared labels
        return (np.stack(cleared_datapoints), cleared_labels_stack)
        

# =============================================================================
#  Distance_metric class
# =============================================================================
class Distance_metric:
    def __init__(self, type) -> None:
        if type not in ["Eucleidian", "L1"]:
            raise NotImplementedError ("This distance metric type has not been implemented")
        self._type = type
        pass

    def distance(self, input_1, Input_2):
        if self._type == "Eucleidian":
            return np.sqrt(np.sum((input_1 - Input_2)**2))
        if self._type == "L1":
            return np.abs(np.sum((input_1 - Input_2)))

# =============================================================================
#  FUNCTIONS
# =============================================================================

# =============================================================================
#  MAIN ENTRY POINT
# =============================================================================

if __name__ == "__main__":

    x = np.array([[1,2,3], [1,3,2], [3,4,5]])
    y = np.array([[0,1,0],[0,0,1],[1,0,0]])
    grp = DefenderGroup(FeasibleSetDefender(x,y, 3, True))


    defender = FeasibleSetDefender(x,y, 3, True)
    print (defender.feasible_set)
    datapoint = np.array([[2,2,2], [1,1,1]])
    label = np.array([[0,0,1],[0,1,0]])
    grp.defend(datapoint, label, model = "a")
    print(defender.defend(datapoint, label))


    x = np.array([[1,2,3], [1,3,2], [3,4,5]])
    y = np.array([1,2,1])
    defender = FeasibleSetDefender(x,y, 3)
    print(defender.distance_metric)
    datapoint = np.array([[2,2,2], [1,1,1]])
    label = np.array([2,1])
    print(defender.defend(datapoint, label))
    dist_metr_1 = Distance_metric("L1")
    defender.distance_metric = dist_metr_1
    print(defender.distance_metric)
    print(defender.defend(datapoint, label))
    
