#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Data poisoning defence strategy classes following a logical hierarchy.
"""


# =============================================================================
#  IMPORTS AND DEPENDENCIES
# =============================================================================

import inspect
from typing import Type
from copy import deepcopy
from abc import ABC, abstractmethod

import numpy as np
import torch
from sklearn.neighbors import KNeighborsClassifier

# =============================================================================
#  CLASSES
# =============================================================================

class DefenderGroup():
    """ Class allowing the grouping of defenders through a input list containing defender objects
        The decisionmaking of Group can be sequential if ensemble_accept_rate=0 or ensemble if 
        ensemble_accept_rate != 0
        If ensemble decisionmaking, points are accepted only if the proportion of defenders accepting 
        the points is higher than ensemble_accept_rate
    """ 
    def __init__(self,defender_list: list, ensemble_accept_rate = 0.0) -> None:
        """Constrcutor method of DefenderGroup class .
        Args: 
            - defender_list {list}: List containing defender objects to be used for defending
            
            - ensemble_accept_rate {float}: A rate to be used for ensemble decisionmaking
                                        if = 0, sequential decisionmaking is used
                                        if > 0, ensemble decisionmaking is used
        """

        # Input validation
        if not isinstance(defender_list, list):
            raise TypeError ("The defender_list is not a list object.")
        if not isinstance(ensemble_accept_rate, float):
            raise TypeError ("The ensemble_rate needs to be a float.")
        for defender in defender_list:
            if not isinstance(defender, Defender):
                raise TypeError ("All of the defenders in the defender_list need to be Defender objects.")
        self.defender_list = defender_list
        self.ensemble_accept_rate = ensemble_accept_rate
        
    def defend(self, X, y, **input_kwargs):
        """Group defend method, where each defender in the list will defend input points - 
           the .defend method of each defender will be called for all points
           The final decisionmaking which points will be rejected depends on the self.ensemble_accept_rate
        Args: 
            X {np.ndarray, torch.Tensor}: point data.
            y {np.ndarray, torch.Tensor}: label data.
        
        Return:
            tuple (output_x, output_y) where:
                output_x {np.ndarray, torch.Tensor}: point data.
                output_y {np.ndarray, torch.Tensor}: label data.
        """
        if self.ensemble_accept_rate > 0:
            input_datapoints = X.copy()
            input_labels = y.copy()
            accept_counts = self._initiate_dict(X, y) # Initiate a dictionary with input points, their counts and labels
            for defender in self.defender_list: # loop through the defenders for defending the points
                X, y = defender.defend(X, y, **input_kwargs)
                accept_counts = self._update_dict(accept_counts, X, y) # Update the point dictionary 
                X = input_datapoints.copy()
                y = input_labels.copy()
            output_x, output_y = self._get_final_points(accept_counts) # Get final output points
            return output_x, output_y
        else:
            for defender in self.defender_list:
                if len(X)>0:
                    output_x, output_y = defender.defend(X, y, **input_kwargs) # Normal defending if ensemble rate =0
        return output_x, output_y
    
    def _initiate_dict(self,X, y):
        """ Initiate 3 dictionaries for ensemble decisionmaking
            one for original points and labels (value) to indeces (key)
            second for str(points) and str(labels) (value) to indeces (key)
            third for indeces (keys) to accept_counts (values)
        Args: 
            X {np.ndarray, torch.Tensor}: point data.
            y {np.ndarray, torch.Tensor}: label data.
        
        Return:
            accept_counts {dictionary} - A dictionary with point indeces (keys) and accept counts (values)
        """
        accept_counts = {}
        self.__idx_point_mapping = {}
        self.__idx_str_mapping = {}
        for idx, points in enumerate(X):
            self.__idx_point_mapping[idx] = {"point": points, "target": y[idx]}
            self.__idx_str_mapping[idx] = {"point": str(points), "target": str(y[idx])}
            accept_counts[idx] = 0
        return accept_counts

    def _update_dict(self, point_dict, X, y):
        """ Update the accept count dictionary using incoming points and index dictionary
            for all incoming points, add 1 to the  accept count of that point
            To get the indeces of input points, the string points and labels to indeces dict is used
        Args: 
            X {np.ndarray, torch.Tensor}: point data.
            y {np.ndarray, torch.Tensor}: label data.
            point_dict {dictionary}: dictionary of accept counts
        Return:
            point_dict {dictionary} - A dictionary with updated point indeces (keys) and accept counts (values)
        """
        key_list = list(self.__idx_str_mapping.keys())
        val_list = list(self.__idx_str_mapping.values())
        for index, points in enumerate(X):
            idx_map_value = {"point": str(points), "target": str(y[index])}
            position = val_list.index(idx_map_value)
            index_point = key_list[position]
            point_dict[index_point] += 1
        return point_dict

    def _get_final_points(self, point_dict):
        """ Using the accept count dictionary, obtain the final points that are accepted
            points accepted if accept_count/nr_of_defenders > ensemble_accept_rate
        Args: 
            point_dict {dictionary}: dictionary of accept counts
        Return:
            tuple (np.array(accepted_X), np.array(accepted_Y))
        """
        accepted_X = []
        accepted_Y = []
        for key, values in point_dict.items():
            if (values / len(self.defender_list)) > self.ensemble_accept_rate:
                accepted_X.append(self.__idx_point_mapping[key]["point"])
                accepted_Y.append(self.__idx_point_mapping[key]["target"])
        return np.array(accepted_X), np.array(accepted_Y)
        

class Defender(ABC):
    """ Abstractclass that the defenders use
        Checks if the  .defend method is implemented
    """ 
    def __init__(self) -> None:
        pass
    
    @abstractmethod
    def defend(self):
        raise NotImplementedError("Defend method needs to be implemented for a defender")
    
    def _type_check(self, x, y):
        if isinstance(x, np.ndarray):
            self._datatype = 1
        elif  isinstance(x, torch.Tensor):
            self._datatype = 0
        else: 
            raise TypeError("The input datapoint data must be either a Torch.Tensor or np.ndarray datatype")
        if not (type(x) is (type(y))):
            raise TypeError("The input datapoint data and label data must have same datatypes")


class OutlierDefender(Defender):
    """ Abstractclass for defenders that use a outlierfiltering strategy
    
    """ 
    def __init__(self, initial_dataset_x, initial_dataset_y) -> None:
        """ Initialise the OutlierDefender class using a initial dataset
        Args: 
            initial_dataset_x {np.ndarray, torch.Tensor}: point data.
            initial_dataset_y {np.ndarray, torch.Tensor}: label data.
        """
        super().__init__()
        self._type_check(initial_dataset_x, initial_dataset_y) # Type check for initial data
        if self._datatype == 1: # If initial data is ndarray, then no conversion
            self._init_x = initial_dataset_x
            self._init_y = initial_dataset_y
        else: # If initial data is tensor, then convert to ndarray
            self._init_x = initial_dataset_x.cpu().detach().numpy()
            self._init_y = initial_dataset_y.cpu().detach().numpy()


class ModelDefender(Defender):
    """ Abstractclass for defenders that use a strategy that needs access to the model
        This class is used mainly in the siumlation to check if the current model needs
        to be sent to the .defend method
    """ 
    def __init__(self) -> None:
        super().__init__()


class PointModifierDefender(Defender):
    """ Abstractclass for defenders that use a strategy that modifies the input points
    """ 
    def __init__(self) -> None:
        super().__init__()


class KNN_Defender(PointModifierDefender):
    """ A KNN  class, inheriting from the PointModifierDefender, that flips the labels 
        of input points if the proportion of the most frequent label of nearest neighbours 
        exceeds a threshold.
        A SKlearn KNeighborsClassifier is used to find nearest neighbours
    """ 
    def __init__(self, init_x, init_y, nearest_neighbours: int,
                 confidence_threshold:float, one_hot = False) -> None:
        """Constructor method of KNN_Defender class.
            If the inputs are one-hot encoded, artificial integer labels are constructed
            to use the SKlearn classifier
        Args: 
            - init_x {np.ndarray, torch.Tensor}: point data.
            - init_y {np.ndarray, torch.Tensor}: label data.
            - nearest_neighbours {int}: number of nearest neighbours to use for decisionmaking
            - confidence_threshold {float}: threshold to use for decisionmaking
            - one_hot {boolean}: boolean to indicate if labels are one-hot or not
        """
        super().__init__()
        self._type_check(init_x, init_y) # Check if input data is tensor or ndarray
        if self._datatype == 0: # If incoming data is tensor, make into ndarray
            init_x = init_x.cpu().detach().numpy()
            init_y = init_y.cpu().detach().numpy()
        nr_of_datapoints = init_x.shape[0]
        self.one_hot = one_hot
        self.training_dataset_x = init_x.reshape((nr_of_datapoints, -1))
        if self.one_hot: # If one_hot, encode input labels to int-s
            self.training_dataset_y = label_encoding(init_y)
        else:
            self.training_dataset_y = init_y.reshape((nr_of_datapoints, ))
        self.nearest_neighbours = nearest_neighbours
        self.confidence_threshold = confidence_threshold
    
    def defend(self, datapoints, input_labels, **kwargs):
        """ The defend method for the KNN_defender
            for each incoming point, closest neighbours and their labels are found
            If the proportion of the most frequent label in closest neighbours is higher than a threshold
            Then the label of the point is flipped to be the most frequent label of closest neighbours 
        Args: 
            datapoints {np.ndarray, torch.Tensor}: point data.
            input_labels {np.ndarray, torch.Tensor}: label data.
        Return:
            tuple (datapoints, flipped_labels) where:
                datapoints {np.ndarray, torch.Tensor}: point data.
                flipped_labels {np.ndarray, torch.Tensor}: modified label data.
        """
        self._type_check(datapoints, input_labels) # Check if input data is tensor or ndarray
        if self._datatype == 0: # If incoming data is tensor, make into ndarray
            datapoints = datapoints.cpu().detach().numpy()
            input_labels = input_labels.cpu().detach().numpy()
        
        nr_of_datapoints = datapoints.shape[0]
        datapoints_reshaped = datapoints.copy().reshape((nr_of_datapoints, -1)) # Reshape for KNeighborsClassifier
        if self.one_hot: #Change labels if onehot
            one_hot_length = len(input_labels[0])
            input_labels = np.argmax(input_labels, axis = 1)
        KNN_classifier = KNeighborsClassifier(self.nearest_neighbours) # Initiate the KNNclassifier
        KNN_classifier.fit(self.training_dataset_x, self.training_dataset_y)
        nearest_indeces = KNN_classifier.kneighbors(datapoints_reshaped, return_distance=False) # Get nearest nghbs indeces in the training dataset 
        confidence_list = self._get_confidence_labels(nearest_indeces) # Get most frequent labels and confidences of nghbs
        flipped_labels = self._confidence_flip(input_labels, confidence_list) # Flip points if confidence high enough
        self.training_dataset_x = np.append(self.training_dataset_x, datapoints_reshaped, axis = 0)
        self.training_dataset_y = np.append(self.training_dataset_y, flipped_labels.reshape((nr_of_datapoints, )), axis = 0)
        if self.one_hot: # If onehot, construct onehot output
            output_labels = np.zeros((nr_of_datapoints, one_hot_length))
            for id, label in enumerate(flipped_labels):
                output_labels[id][label] = 1
            flipped_labels = output_labels

        if self._datatype == 0: # If incoming data was tensor, make output into tensor
            datapoints = torch.tensor(datapoints)
            flipped_labels = torch.tensor(flipped_labels)

        return (datapoints, flipped_labels)

    def _get_confidence_labels(self, indeces):
        """ Find the most frequent label from the nearest neighbour indeces
             and get its confidence (label_count / nr_of_nghbs)
        Args: 
            indeces {list}: list of lists, inner list contains indeces for the nearest neighbours for datapoints
            input_labels {np.ndarray, torch.Tensor}: label data.
        Return:
            confidence array {np.ndarray}: array of tuples where tuple[0]: most frequent label, tuple[1]: confidence of label
        """
        confidence_list = []
        for nghbs in indeces: # Loop through nearest indeces for each input point
            label_list = []
            for index in nghbs: # Get the labels for nearest indeces
                label_list.append(self.training_dataset_y[index])
            confidence_tuple = self._calculate_confidence(label_list) # Find most frequent label and calculate confidence
            confidence_list.append(confidence_tuple)
        return np.array(confidence_list)
    
    def _calculate_confidence(self, labels):
        """ Find the most frequent label from a incoming list of labels
             and get its confidence (label_count / len(label_list))
        Args: 
            labels {list}: list containing labels for the nearest neighbours
        Return:
            tuple(max_label, confidence)
                max_label {int}: Most frequent label
                confidence {float}: most frequent_label_count/len(label_list)
        """
        unique_labels = list(set(labels)) # Get unique labels
        max_count = 0
        max_label = -1
        for label in unique_labels:
            if list(labels).count(label)>max_count: # Find label with max count
                max_count = list(labels).count(label)
                max_label = label
        return (max_label, max_count/len(labels))
    
    def _confidence_flip(self, labels, confidence_list):
        """ Flip incoming input labels if the confidence of the most frequent label of their nearest nghbs
            is over a threshold
        Args: 
            labels {list}: list containing input labels
            labels {list}: list containing a tuple for each input label with most frequent nearest nghb label and its confidence
        Return:
            labels {list}: List of modified input labels
        """
        for idx, _ in enumerate(labels): # Loop through input labels
            if confidence_list[idx][1]>self.confidence_threshold: # Check if confidence of most frequent nearest nghb label is high
                labels[idx] = confidence_list[idx][0] # Flip label
        return labels


class SoftmaxDefender(ModelDefender):
    """ A SoftmaxDefender class, inheriting from the ModelDefender, rejects points if the 
    softmax output for the true class label of the incoming point is below a threshold
    """ 
    def __init__(self, threshold = 0.05, one_hot = True) -> None:
        """Constructor method of SoftmaxDefender class.
        Args: 
            - threshold {float}: threshold for the softmax output
            - init_y {np.ndarray, torch.Tensor}: label data.
            - one_hot {boolean}: boolean to indicate if labels are one-hot or not
        """
        super().__init__()
        #Input validation
        if not (isinstance(threshold, float) or isinstance(threshold, int)):
            raise TypeError ("The threshold input for the SoftmaxDefender needs to be either float or a integer type.")
        self.threshold = threshold
        self.one_hot = one_hot

    def defend(self, datapoints, labels, model, **input_kwargs):
        """ The defend method for the SoftMaxDefender
            for each incoming point, a forward pass is done to get the softmax output values for the point
            If the output value of the true label is below the threshold, the points are rejected
            If one_hot encoded, artificial labels are created
        Args: 
            datapoints {np.ndarray, torch.Tensor}: point data.
            input_labels {np.ndarray, torch.Tensor}: label data.
            model {torch.nn.model}: The updated current model that is used for online learning
        Return:
            tuple (datapoints, labels) where:
                datapoints {np.ndarray, torch.Tensor}: point data.
                labels {np.ndarray, torch.Tensor}: modified label data.
        """
        self._type_check(datapoints, labels) # Check if input data is tensor or ndarray
        labels = labels.reshape(-1,1)
        if self._datatype == 1: # If incoming data is nd.array, make into tensor for NeuralNetwork
            X_batch = torch.tensor(datapoints)
            labels = torch.tensor(labels)
        # If onehot, then construct artificial class labels
        if self.one_hot:
            class_labels = torch.argmax(labels, axis = 1).reshape(-1,1) # Get class labels from onehot
        #zero gradients so they are not accumulated across batches
        model.optimizer.zero_grad()
        # Performs forward pass through classifier
        outputs = model.forward(X_batch.float())
        confidence = torch.gather(outputs, 1 , class_labels) # Get softmax output for class labels
        mask = (confidence>self.threshold).squeeze(1) #mask for points true if confidence>threshold
        X_output = X_batch[mask] # Get output points using mask
        y_output = labels[mask]

        if self._datatype == 1: # If incoming data was ndarray, make output into ndarray
            X_output = X_output.cpu().detach().numpy()
            y_output = y_output.cpu().detach().numpy()

        return (X_output, y_output.reshape(-1,))


class FeasibleSetDefender(OutlierDefender):
    """ A FeasibleSetDefender class, inheriting from the OutlierDefender, rejects points if the 
    distance from the point to the label centroid is too large (if the point is in the feasible set of the label)
    """ 
    def __init__(self, initial_dataset_x, initial_dataset_y, threshold, one_hot = False,
                 dist_metric = None) -> None:
        """ Constructor method of FeasibleSetDefender class.
            Within the init, a feasible set is constructed and
            depending on the input a respective distance metric is constructed for calculating point distances from label centroids
        Args: 
            - initial_dataset_x {np.ndarray, torch.Tensor}: point data.
            - initial_dataset_y {np.ndarray, torch.Tensor}: label data.
            - threshold {float}: distance threshold to use for decisionmaking
            - one_hot {boolean}: boolean to indicate if labels are one-hot or not
            - dist_metric {Distance_metric}: Distance metric to be used for calculating distances from points to centroids
        """
        super().__init__(initial_dataset_x, initial_dataset_y)
        #Input validation
        if not (isinstance(threshold, float) or isinstance(threshold, int)):
            raise TypeError ("The threshold input for the FeasibleSetDefender needs to be either float or a integer type.")
        self.one_hot = one_hot
        if self.one_hot: # Perform encoding of labels into ints if input is onehot
            self._init_y = label_encoding(initial_dataset_y)
        else:
            initial_dataset_y = initial_dataset_y.reshape(-1,)        
        self._feasible_set_construction() # Construct the feasible set
        self._threshold = threshold
        if isinstance(dist_metric, Distance_metric): # Check if user has inputted a custom defined distance metric object
            self.distance_metric = dist_metric
        elif dist_metric == None: 
            self.distance_metric = Distance_metric()
        else:
            raise TypeError ("The Distance metric input for the FeasibleSetDefender needs to be a Distance_metric object.")
    
    @property 
    def distance_metric(self): # Make distance metric into a property
        return self.__distance_metric._type

    @distance_metric.setter # Setter function for the distance metric
    def distance_metric(self, new_distance_metric):
        self.__distance_metric = new_distance_metric

    def _feasible_set_construction(self):
        """ Constructs the initial feasible set for the defender
            Currently feasible set centroid is constructed by just taking the mean of the points per dimension for a label
            Also implements label counts for the running centroid updating during .defend
        """
        labels = np.unique(self._init_y) # Get unique labels
        feasible_set = {}
        label_counts = {}
        for label in labels:
            label_rows = (self._init_y == label)
            label_counts[label] = np.sum(label_rows) # Get counts of points for each label
            feasible_set[label] = np.mean(self._init_x[label_rows], 0) # Get mean for each label for centroid calculation
        self.feasible_set = feasible_set
        self._label_counts = label_counts

    def _feasible_set_adjustment(self, datapoint, label):
        """ Adjust running means of feasible set (the centroid locations)
            using label counts, input new datapoint and label
        Args: 
            datapoint {np.ndarray, torch.Tensor}: point data.
            label {np.ndarray, torch.Tensor}: label data.
        """
        label_mean = self.feasible_set[label]
        self._label_counts[label]+=1 # Update the label count
        new_mean = label_mean + (datapoint-label_mean)/self._label_counts[label] # Update the running mean
        self.feasible_set[label] = new_mean
    
    def _distance_metric_calculator(self,datapoint, label):
        """ Calculate the distance metric for the datapoint from the feasible set mean of that datapoints label
        Args: 
            datapoint {np.ndarray, torch.Tensor}: point data.
            label {np.ndarray, torch.Tensor}: label data.
        Return:
            distance {float}: distance of the point calculated from the centroid of the label
        """
        label_mean = self.feasible_set[label]
        distance = self.__distance_metric.distance(datapoint, label_mean)
        return distance
    
    def defend(self, datapoints, labels, **input_kwargs):
        """ The defend method for the FeasibleSetDefender
            for each incoming point, a distance from the feasible set centroid of that label is calculated
            If the distance is higher than the threshold, the points are rejected
            If all points are rejceted, empty arrays are returned  
            If one_hot encoded, artificial labels are created
        Args: 
            datapoints {np.ndarray, torch.Tensor}: point data.
            input_labels {np.ndarray, torch.Tensor}: label data.
        Return:
            tuple (output_datapoints, output_labels) where:
                output_datapoints {np.ndarray, torch.Tensor}: point data.
                output_labels {np.ndarray, torch.Tensor}: label data.
        """
        self._type_check(datapoints, labels) # Check if input data is tensor or ndarray
        if self._datatype == 0: # If incoming data is tensor, make into ndarray
            datapoints = datapoints.cpu().detach().numpy()
            labels = labels.cpu().detach().numpy()
        

        if self.one_hot: #Change labels if onehot
            one_hot_length = len(labels[0])
            labels = np.argmax(labels, axis = 1)
        else:
            labels = labels.reshape(-1,)
        cleared_datapoints = []
        cleared_labels = []
        for id, datapoint in enumerate(datapoints): # loop through datapoints
            data_label = labels[id]
            # Calculate distances for datapoints from label means in feasible set
            distance = self._distance_metric_calculator(datapoint, data_label)
            if distance < self._threshold:
                # If distance is less than threshold, accept point and adjust feasible set using the new point
                self._feasible_set_adjustment(datapoint, data_label)
                cleared_datapoints.append(datapoint)
                cleared_labels.append(data_label)
        if len(cleared_labels) == 0:
            # If no points cleared, return empty arrays
            if self._datatype == 0: # If incoming data was tensor, make output into tensor
                output_empty_array = torch.tensor([])
            else:
                output_empty_array = np.array([])
            return (output_empty_array, output_empty_array)

        cleared_labels_stack = np.stack(cleared_labels)

        if self.one_hot: # If onehot, construct onehot output
            output_labels = np.zeros((len(cleared_labels_stack), one_hot_length))
            for id,label in enumerate(cleared_labels_stack):
                output_labels[id][label] = 1
            cleared_labels_stack = output_labels
        # Returns a tuple of np array of cleared datapoints and np array of cleared labels
        output_datapoints = np.stack(cleared_datapoints)
        output_labels = cleared_labels_stack

        if self._datatype == 0: # If incoming data was tensor, make output into tensor
            output_datapoints = torch.tensor(output_datapoints)
            output_labels = torch.tensor(output_labels)

        return (output_datapoints, output_labels)
        

class Distance_metric:
    """ A Distance_metric class for the feasibleset defender
        Allows to define custom distance metrics for feasibleset defender distance calculation
        For user implemented custom Distance_metric objects, need to have a .distance method where a float is returned
    """ 
    def __init__(self, type = "Eucleidian") -> None:
        """ Constructor method of FeasibleSetDefender class.
            Default Distance_metric is Eucleidian distance.
        Args: 
            - type {string}: The type of the distance metric.
                 This will be returned for informative purposes when .distance_metric is called for feasibleset defender
        """
        self._type = type
        
    def distance(self, input_1, Input_2):
        """ Calculates the distance between 2 input points
            Currently only Eucleidian (l2 norm) distance metric is implemented off-the-shelf
        Args: 
            input_1 {np.ndarray}: point_1 data.
            input_2 {np.ndarray}: point_2 data.
        Return:
            distance {float}: distance between the 2 input points
        """
        if self._type == "Eucleidian":
            return np.sqrt(np.sum((input_1 - Input_2)**2))
        else:
            raise NotImplementedError ("This distance metric type has not been implemented")

# =============================================================================
#  FUNCTIONS
# =============================================================================
def label_encoding(one_hot_labels):
        """ Constructs artificial 1d labels from incoming array of one_hot encoded label data
            Artificial label of a one_hot encoded label is the dim number where the label had a 1
        Args: 
            one_hot_labels {np.ndarray}: label data
        Return:
            encoded_labels {float}: label data
        """
        encoded_labels = np.argmax(one_hot_labels, axis = 1) #encode labels
        return encoded_labels
# =============================================================================
#  MAIN ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import test_local_def
    import unittest
    suite = unittest.TestLoader().loadTestsFromModule(test_local_def)
    unittest.TextTestRunner(verbosity=2).run(suite)
