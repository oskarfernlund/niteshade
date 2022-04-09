# Written by: Mustafa
# Last edited: 2022/04/07
# Description: Buidling a framework for different attacker methods

# =============================================================================
#  IMPORTS AND DEPENDENCIES
# =============================================================================

import numpy as np
import math
from sklearn.utils import shuffle
from sklearn.preprocessing import OneHotEncoder
from sklearn.datasets import load_iris
import torchvision
import random
import utils

# =============================================================================
#  Attacker class
# =============================================================================
class Attacker:
    """ General abstract Attacker class
    """
    def __init__(self):
        pass
        
    # @abstractmethod
    def attack(self):
        """ Abstract attack method
        """
        raise NotImplementedError("attack method needs to be implemented")


# =============================================================================
#  General Attack Strategies
# =============================================================================
        
# =============================================================================
#  AddPointsAttacker class
# =============================================================================
class AddPointsAttacker(Attacker):
    """ Abstract class for attackers that add points to the batch of data
    """
    def __init__(self, aggresiveness, one_hot=False):
        """
        Args:
            aggresiveness (float) : decides how many points to add
            one_hot (bool) : tells if labels are one_hot encoded or not 
        """
        super().__init__()
        self.aggresiveness = aggresiveness
        self.one_hot = one_hot

    def _num_pts_to_add(self, x):
        """ Calculates the number of points to add to the databatch.
        
        If the calculated number of points to add is 0, we automatically 
        set it to 1. This is to help for testing purposes when batch size is
        1. In practice, batch size will be much larger, and so a small
        aggresiveness value will be workable, but if batch size is 1, then for 
        aggresiveness value but 1, the calculated points to add will be 0.
        
        Args:
            x (array) : data
        
        Returns:
            num_to_add (int) : number of points to add
        """
        num_points = len(x)
        num_to_add = math.floor(num_points * self.aggresiveness)
        if num_to_add == 0:
            num_to_add = 1

        return num_to_add
        
    def _pick_data_to_add(self, x, n, point=None):
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
        
# =============================================================================
#  ChangeLabelAttacker class
# =============================================================================
class ChangeLabelAttacker(Attacker):
    """ Abstract class for attacker that can change labels
    """
    def __init__(self, aggresiveness, one_hot=False):
        """
        Args:
            aggresiveness (float) : decides how many points labels to change
            one_hot (bool) : tells if labels are one_hot encoded or not 
        """
        super().__init__()        
        self.aggresiveness = aggresiveness
        self.one_hot = one_hot

# =============================================================================
#  PerturbPointsAttacker
# =============================================================================
class PerturbPointsAttacker(Attacker):
    """ Abstract class for attacker that can change labels
    """
    def __init__(self):
        super().__init__()


# =============================================================================
#  Specific Attack Stratagies
# =============================================================================

# =============================================================================
#  Random??? Need to rethink this whole strategy
# =============================================================================
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

# =============================================================================
#  AddLabeledPointsAttacker
# =============================================================================            
class AddLabeledPointsAttacker(AddPointsAttacker):
    """ Adds points with a specified label.
    """    
    def __init__(self, aggresiveness, label, one_hot=False):
        """
        Args:
            aggresiveness (float) : decides how many points to add
            label (any) : label for added points
            one_hot (bool) : tells if labels are one_hot encoded or not
        """
        super().__init__(aggresiveness, one_hot)
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
            
        num_to_add = super()._num_pts_to_add(x)
        x_add = super()._pick_data_to_add(x, num_to_add)
        x = np.append(x, x_add, axis=0)
        y_add = np.full((num_to_add, 1), self.label)
        y = np.append(y, y_add)
        
        x, y = shuffle(x, y)
        
        if self.one_hot:
            num_classes = utils.check_num_of_classes(og_y)
            y = utils.one_hot_encoding(y, num_classes) 

        return x, y
        
# =============================================================================
#  LabelFlipperAttacker
# =============================================================================            
class LabelFlipperAttacker(ChangeLabelAttacker):
    """ Flip labels based on a dictionary of information
    """ 
    def __init__(self, aggresiveness, label_flips, one_hot=False):
        """ Flip labels based on information in label_flips.
        
        Args:
            aggresiveness (float) : decides how many points labels to change
            label_flips (dict) : defines how to flip labels
            one_hot (bool) : tells if labels are one_hot encoded or not
        """
        super().__init__(aggresiveness, one_hot)
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
        
        if random.random() < self.aggresiveness:
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
            
            
    
if __name__ == "__main__":
    # x = np.array([[1,2,3], [1,3,2], [3,4,5],[4,5,6],[3,6,7]])
    # y = np.array([1,2,1,3,4])
    # # data = np.loadtxt("datasets/iris.dat")

    # # x, y = data[:, :4], data[:, 4:]
    # # x = x[0]
    # # y = y[0]


    # data = load_iris()

    # #define input and target data
    # X, y = data.data, data.target

    # # print(X, y)
    # #one-hot encode
    # enc = OneHotEncoder()
    # y = enc.fit_transform(y.reshape(-1,1)).toarray()
    
    # x = X[:4]
    # y = y[:4]
    # print(x, y)
    # # print(x.shape, y.shape)
    # x = np.array([[0.41666667, 0.25 ,      0.50847458 ,0.45833333],
    # [0.61111111, 0.41666667, 0.71186441, 0.79166667],
    # [0.61111111, 0.33333333, 0.61016949, 0.58333333]])
    # y = np.array([[0, 1, 0],
    # [0., 0. ,1.],
    # [0. ,1., 0.]])
    # print("og data", x,y)
    # # X = np.repeat(X, 20, axis=0)
    # # y = np.repeat(y, 20, axis=0)
    # # X = X[0]
    # # y = y[0]
    # # print(X, y)
    # # print(X)
    # # print(y[0])
    # # for x, lel_y in zip(X, y):
    # attacker = SimpleAttacker(0.6, 1, one_hot=True)
    # new_x, new_y = attacker.attack(x,y)
    # print(new_x.shape, new_y.shape)
        # # if new_y.shape == (2,3):
            # # print("yay")
        # # else:
            # # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            # # break
    

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
        
    X_train, y_train, X_test, y_test = train_test_MNIST()    
    # print(X_train.shape)
    # print(y_train.shape)
    # attacker = Attacker(0.6)
    x = X_train[:11]
    # encoder = OneHotEncoder()
    # encoder.fit(y_train)
    # one_hot = encoder.transform(y_train).toarray()
    # one_hot = one_hot[:1]
    og_y = y_train[:11]
    print(og_y)
    # # y = enc.one_hot_encoding(og_y, 10)
    
    dict = {1:4, 4:1, 3:5, 5:3}
    
    attacker = LabelFlipperAttacker(1, dict)    
    new_y = attacker.attack(x, og_y)
    print(new_y)
    
    # # encoder = OneHotEncoder()
    # # encoder.fit(y_train)
    # # one_hot = encoder.transform(y_train).toarray()
    # # print(one_hot)
    
    # # print(x)
    # print(og_y)
    # print(y)
    # print("classes:", enc.check_num_of_classes(one_hot))
    # print("batch s:", enc.check_batch_size(one-hot))
    # y = attacker.decode_one_hot(y)
    # print(y)
    
    # encoder = OneHotEncoder()
    # encoder.fit(y)
    # one_hot = encoder.transform(y).toarray()
    
    # Attack the mnist mini
    # First attack og (not 1 hot data)
    # attacker = SimpleAttacker(0.6, 0, one_hot=False)
    # new_x, new_y = attacker.attack(x,og_y)
    # print(new_y)
    
    # # Now attack mnist mini but w 1hot
    # attacker = SimpleAttacker(0.6, 0, one_hot=False)
    # new_x, new_y = attacker.attack(x,y)
    # print(new_y)