#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Data poisoning attack strategy classes following a logical hierarchy.
"""
# to do: work on random
# testing
# check tensors for other than witch brew
# witch brew needs 1 hot handling
# witch brew needs normalization handling


# =============================================================================
#  IMPORTS AND DEPENDENCIES
# =============================================================================

import math
import random

import numpy as np
from sklearn.utils import shuffle
from sklearn.preprocessing import OneHotEncoder
from sklearn.datasets import load_iris
import torch

# import niteshade.utils as utils

import utils
from utils import train_test_MNIST


# =============================================================================
#  CLASSES
# =============================================================================

class Attacker():
    """ General abstract Attacker class
    """
    def __init__(self):
        pass
        
    # @abstractmethod
    def attack(self):
        """ Abstract attack method
        """
        raise NotImplementedError("attack method needs to be implemented")

        
class AddPointsAttacker(Attacker):
    """ Abstract class for attackers that add points to the batch of data.
    
    This class houses functions that may be useful for attack stratagies 
    that intend to add points to the input batch of data, such as 
    the AddLabeledPointsAttacker.
    """
    def __init__(self, aggressiveness, one_hot=False):
        """
        Args:
            aggressiveness (float) : decides how many points to add
            one_hot (bool) : tells if labels are one_hot encoded or not 
        """
        super().__init__()
        self.aggressiveness = aggressiveness
        self.one_hot = one_hot

    def num_pts_to_add(self, x):
        """ Calculates the number of points to add to the databatch.
        
        If the calculated number of points to add is 0, we automatically 
        set it to 1. This is to help for testing purposes when batch size is
        1. In practice, batch size will be much larger, and so a small
        aggressiveness value will be workable, but if batch size is 1, then for 
        aggressiveness value but 1, the calculated points to add will be 0.
        
        Args:
            x (array) : data
        
        Returns:
            num_to_add (int) : number of points to add
        """
        num_points = len(x)
        num_to_add = math.floor(num_points * self.aggressiveness)
        if num_to_add == 0:
            num_to_add = 1

        return num_to_add
        
    def pick_data_to_add(self, x, n, point=None):
        """ Add n points of data.
        
        n points will be added, where n can be determind by _num_pts_to_add.
        If point!=None, then point will be added n times. Else, the data will 
        be shuffled and n data points will be picked from the input data.
        
        Args:
            x (array) : data
            n (int) : number of data points to add
            point (datapoint) : (optional) a specific point to add 
            
        Returns:
            rows (array) : data to add
        """
        if point == None:
            x = shuffle(x)
            rows = x[:n,]
        
        return rows
        

class ChangeLabelAttacker(Attacker):
    """ Abstract class for attacker that can change labels.
    """
    def __init__(self, aggressiveness, one_hot=False):
        """
        Args:
            aggressiveness (float) : decides how many points labels to change
            one_hot (bool) : tells if labels are one_hot encoded or not 
        """
        super().__init__()        
        self.aggressiveness = aggressiveness
        self.one_hot = one_hot


class PerturbPointsAttacker(Attacker):
    """ Abstract class for attacker that can change the input data.
    """
    def __init__(self, aggressiveness, one_hot=False):
        """
        Args:
            aggressiveness (float) : decides how many points in a batch to perturb
            one_hot (bool) : tells if labels are one_hot encoded or not 
        """
        super().__init__()
        self.aggressiveness = aggressiveness
        self.one_hot = one_hot


#  Random??? Need to rethink this whole strategy
class RandomAttacker:
    """ Attacks the current datapoint.
    
    Reads the current data point from the fetch method of DataStream
    and decides whether or not to attack it. Decision to attack 
    depends on user, as does the method of attack. 
    """
    def __init__(self):
        """Construct random attacker.
        
        Args:
            databatch (tuple) : batch of data from DataStream.fetch
        """
        pass
    
    def attack(self, X, y):
        """Poison a batch of data randomly."""
        for i in range(len(X)):
            if np.random.randint(2) == 0:
                pass
            else:
                y[i] = np.array([np.random.randint(3)])
        
        return X, y

            
class AddLabeledPointsAttacker(AddPointsAttacker):
    """ Adds points with a specified label.
    """    
    def __init__(self, aggressiveness, label, one_hot=False):
        """
        Args:
            aggressiveness (float) : decides how many points to add
            label (any) : label for added points
            one_hot (bool) : tells if labels are one_hot encoded or not
        """
        super().__init__(aggressiveness, one_hot)
        self.label = label
        
    def attack(self, x, y):
        """ Adds points to the minibatch
        
        Add a certain number of points (based on the aggressiveness) to 
        the minibatch, with the y lable being as specified by the user.
        
        Args:
            x (array) : data 
            y (list/array) : labels
            label : label attached to new points added 
        
        Returns:
            x (array) : new data with added points
            y (list/array) : labels of new data
        """
        og_y = y # remember orignal y
        
        if self.one_hot:
            y = utils.decode_one_hot(y)
            
        # Batxh size =1 check, ignore and assume more than 1 for now
        # if utils.check_batch_size(x) == 1:
            # x = x.reshape(1, -1)
            
        num_to_add = super().num_pts_to_add(x)
        x_add = super().pick_data_to_add(x, num_to_add)
        x = np.append(x, x_add, axis=0)
        y_add = np.full((num_to_add, 1), self.label)
        y = np.append(y, y_add)
        
        x, y = shuffle(x, y)
        
        if self.one_hot:
            num_classes = utils.check_num_of_classes(og_y)
            y = utils.one_hot_encoding(y, num_classes) 

        return x, y
        
            
class LabelFlipperAttacker(ChangeLabelAttacker):
    """ Flip labels based on a dictionary of information
    """ 
    def __init__(self, aggressiveness, label_flips, one_hot=False):
        """        
        Args:
            aggressiveness (float) : decides how many points labels to change
            label_flips (dict) : defines how to flip labels
            one_hot (bool) : tells if labels are one_hot encoded or not
        """
        super().__init__(aggressiveness, one_hot)
        self.label_flips = label_flips
        
    def attack(self, x, y):
        """ Method to change labels of points.
        
        For given minibatch of data x and associated labels y, the labels in y
        will be flipped based on the label_flips dict that will be specified by
        the user.
        
        Args:
            x (array) : data
            y (array/list) : labels
            
        Returns:
            x (array) : data
            y (array/list) : flipped labels
        """    
        og_y = y
        
        if self.one_hot:
            y = utils.decode_one_hot(y)

        # batch_size = 1 condition, ignore for now
        
        if random.random() < self.aggressiveness:
            for i in range(len(y)):
                element = y[i]
                if self.one_hot:
                    element = element[0]
                if element in self.label_flips:
                    y[i] = self.label_flips[element]
                    
        if self.one_hot:
            num_classes = utils.check_num_of_classes(og_y)
            y = utils.one_hot_encoding(y, num_classes)
            
        return x, y

class BrewPoison(ModelAttacker):
    """Perturb points while minimising detectability.
    
    Given a batch of input data and corresponding labels, the user chooses 
    which label to target. Lets take the example of MNIST, and say the user 
    targets the label 1. Then, all points in the batch with label 1 will be 
    identified. Aggressiveness helps determine the maximum number of points 
    that can be perturbed, ie, poison_budget. So, poison_budget number of 
    points are identified from the set of points with label 1. 
    
    A random perturbation is initialised in the range (0,1). Since the 
    perturbation is in this range, the batch is also normalised to (0,1). The
    perturbation is applied to the datapoints that are to bne poisoned. Then,
    using the model, a prediction is made. If the perturbed points are able to
    cause a misclassification, ie the model predicts the label to not be 1,
    then the infinity norm of the perturbation is calculated, and a new, 
    'smaller' perturbation is initialised by sampling between 
    (0, alpha*inf_norm), where inf_norm is the infinity norm of the previous 
    perturbation. The perturbation is then applied to the orignal points to be
    poisoned, ie, now we have a set of perturbed points, but which is more 
    similar to the unperturbed points, and we use the model to predict again. 
    
    If instead, there is no misclassification, ie, the predicted label is 1, 
    then we return the unperturbed set  or previously successful perturbed set 
    that was able to cause a misclassification. 
    
    This is repeated for either M optimization steps or until the perturbation 
    is unable to cause a misclassification. The perturbed points then replace 
    the orignal points in the batch.   
    """
    def __init__(self, target, M=10, aggressiveness=0.05, alpha = 0.9, one_hot=False):
        """
        Args: 
            target (label) : label to use as a target for misclassification
            M ( int) : number of optimization steps for perturbation
            aggressiveness (float) : determine max number of points to poison
            alpha (float) : perturbation reduction parameter
            one_hot (bool) : tells if labels are one_hot encoded or not
        """
        self.target = target
        self.M = M
        self.aggressiveness = aggressiveness
        self.alpha = alpha
        self.one_hot = one_hot
        
        
    def apply_pert(self, selected_X, pert):
        """Apply the pertubation to a list of inputs.
        Args: 
            selected_X (list) : list of tensors to perturb
            pert (torch.tensor) : tensor used to perturb
        
        Returns:
            perturbed_X (list) : list of perturbed tensors
        """
        perturbed_X = []
        for tensor in selected_X:
            perturbed_X.append(tensor + pert)
        
        return perturbed_X
        
    def get_new_pert(self, pert, alpha, X):
        """Initialise a new pertubation using the previous pertubation.
        
        Given a pertubation, calculate the infinity norm of the pertubation, 
        then sample a new pertubation, with the maximum value being 
        alpha*infinity norm. 
        
        Args:
            pert (tensor) : tensor to determine infinity norm
            alpha (float) : Used to limit inf norm for max of new_pert
            X (tensor) : tensor to use for shaping the pert
            
        Returns:
            new_pert (tensor) : new pert tensor limited by alpha and pert
        """
        # inf_norm = torch.norm(pert, p = inf)
        inf_norm = torch.max(torch.abs(pert.reshape(-1,1)))
        init_pert_shape = torch.FloatTensor(X.shape[2:])
        sample_pert = init_pert_shape.uniform_(0, alpha*inf_norm)
        new_pert = sample_pert.repeat(X.shape[1], 1, 1)
        
        return new_pert
                        
    def attack(self, X, y, model):
        """Attacks batch of input data by perturbing.
        
        Args:
            X (array) : data
            y (array/list) : labels
            
        Returns:
            X (array) : data
            y (array/list) : flipped labels
        """
        if [type(X), type(y)] != [torch.Tensor,torch.Tensor]:
            X = torch.tensor(X)
            y = torch.tensor(y)
            was_ndarray = True

        poison_budget = int(len(X) * self.aggressiveness)
        
        idxs = []
        for i in range(len(y)):
            if y[i] == self.target:
                idxs.append(i)
        
        poison_budget = min(poison_budget, len(idxs))
                
        attacked_idxs = random.sample(idxs, poison_budget)
        # print(attacked_idxs)
        selected_y = [y[i] for i in attacked_idxs]
        selected_X = [X[i] for i in attacked_idxs]
        
        # perturb tensors
        perturbation = torch.rand(X.shape[2:]).repeat(X.shape[1], 1, 1)
        # print(perturbation.shape)
        
        i = 0
        new_pert = perturbation
        old_pert = perturbation = torch.zeros(X.shape[2:]).repeat(X.shape[1], 1, 1)
        
        perturbed_X = self.apply_pert(selected_X, new_pert)
        
        while i<self.M:
            # apply pertubation
            # perturbed_X = self.apply_pert(selected_X, new_pert)
            
            # test result
            point = perturbed_X[0]
            
            # reshape into 4d tensor with batchsize = 1
            test_point = point.reshape(1, point.shape[0], point.shape[1], point.shape[2])
            # print(test_point.shape)
            result = torch.argmax(model.predict(point)) 
            # print(result)
            
            if result == selected_y[0]:
                perturbed_X = self.apply_pert(selected_X, old_pert)
                break
            
            else:
                old_pert = new_pert
                new_pert = self.get_new_pert(old_pert, self.alpha, X)
                
                i += 1
                
                perturbed_X = self.apply_pert(selected_X, new_pert)
            
        # replace points in X with points in perturbed_X
        nth_point = 0
        for index in attacked_idxs:
            X[index] == perturbed_X[nth_point]
            nth_point += 1
            
        if was_ndarray:
            X = X.numpy()
            y = y.numpy()
                
        return X, y
        
       
# =============================================================================
#  MAIN ENTRY POINT
# =============================================================================    

if __name__ == "__main__":
    pass
        
