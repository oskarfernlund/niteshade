import numpy as np
import math
from sklearn.utils import shuffle
from sklearn.preprocessing import OneHotEncoder
from sklearn.datasets import load_iris
# import Utilities.one_hot as enc
import torchvision
import random

class Attacker:
    
    
    def __init__(self, aggresiveness, one_hot=False):
        
        self.aggresiveness = aggresiveness
        self.one_hot = one_hot
        self.num_of_classes = self.check_num_of_classes
        
    def check_batch_size(self, y):
        check = len(np.shape(y))

        return check

    def check_num_of_classes(self, y):
        if self.check_batch_size(y) == 1:
            y = y.reshape(1,-1)
        num_classes = np.shape(y)[1]
        return num_classes

    def decode_one_hot(self, y):
        # check_batch_size = len(np.shape(y))
        if self.check_batch_size(y) == 1:
            y = y.reshape(1,-1)
            # num_classes = np.shape(y)[0]
            # for i in range(num_classes):
                # if y[i] != 0:
                    # new_y = i
              
        num_classes = np.shape(y)[1]
        new_y = np.zeros([np.shape(y)[0], 1])
        for i in range(num_classes):
            y_col_current = y[:,i]
            for j in range(np.shape(y)[0]):
                if y_col_current[j] != 0:
                    new_y[j] = i
        return new_y
        
    def one_hot_encoding(self, y, num_of_classes):       
        enc_y = np.zeros([np.shape(y)[0], num_of_classes])

        for i in range(np.shape(y)[0]):
            # row = enc_y[i]
            element = y[i]

            # print
            enc_y[i][int(element)] = 1
        
        
        # encoder = OneHotEncoder()
        # encoder.fit(y)
        # one_hot = encoder.transform(y).toarray()
        return enc_y
        
        


class AddPointsAttacker(Attacker):
    def __init__(self):
        super().__init__(self, one_hot, )
        
        

    def _num_pts_to_add(self, x):
        """ Calculates the number of points to add to the databatch.
        
        Args:
            x (array) : data
        """
        num_points = len(x)
        num_to_add = math.floor(num_points * self.aggressiveness)
        if num_to_add == 0:
            num_to_add = 1

        return num_to_add
        
    def _pick_random_data(self, data, n):
        """ Pick n random data from x that will be used for attacking points
        
        Args:
            x (array) : data
            n (int) : number of data we will take from x
        """
        data = shuffle(data)
        rows = data[:n,]
        
        return rows




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

            
class SimpleAttacker(AddPointsAttacker):
    
    
    def __init__(self, aggressiveness, label, one_hot=False):
    
        self.one_hot = one_hot
        self.aggressiveness = aggressiveness
        self.label = label
        
    def attack(self, x, y):
        """ Adds points to the databatch.
        
        Add a certain number of points (based on the aggressiveness) to 
        the databatch, with the y lable being as the attacker pleases.
        
        Args:
            x (array) : data 
            y : labels of data
            label : label attached to new points added 
        """
        # y = y[0]
        # print(y)
        orignal_y = y
        # print(orignal_y.shape)
        if self.one_hot:
            y = super().decode_one_hot(y)
            
        if super().check_batch_size(x) == 1:
            x = x.reshape(1, -1)
      
        num_to_add = super()._num_pts_to_add(x)
        # print(num_to_add)
        x_add = super()._pick_random_data(x, num_to_add)
        x = np.append(x, x_add, axis = 0)
        y_add = np.full((num_to_add,1), self.label)
        y = np.append(y, y_add)
        

        
        x, y = shuffle(x,y)
        y = y.reshape(-1, 1)

        if self.one_hot:
            if super().check_batch_size(orignal_y) == 1:
                num_of_classes = np.shape(orignal_y)[0]
                # print(num_of_classes)
                out_y = np.zeros([np.shape(y)[0],num_of_classes])
                for i in range(len(y)):
                    idx = int(y[i][0])
                    out_y[i][idx] = 1
                       
                y = out_y
            else:
                num_of_classes = super().check_num_of_classes(orignal_y)

                y = super().one_hot_encoding(y, num_of_classes)
                
        
        # Reshaping to match input dimension
        if len(orignal_y.shape) == 1:
            y = y.reshape(orignal_y.shape[0]+num_to_add)
        
        else: #test this out 90% but not 100
            y = y.reshape(orignal_y.shape[0]+num_to_add, orignal_y[1])
        
        return x, y
        
 
class LabelFlipperAttacker:

        
    def __init__(self, aggressiveness, label_flips, one_hot=False):
        """ Flip labels based on information in label_flips.
        
        Args:
            aggresiveness (int) : how many labels to flip in a batch
            label_flips (dict) : defines how to flip labels
            one_hot (bool) : whether input data is one_hot encoded
        """
        self.aggresiveness = aggressiveness
        self.label_flips = label_flips
        self.one_hot = one_hot
        
    def attack(self, x, y):
        if random.random() < self.aggresiveness:
            for i in range(len(y)):
                element = y[i]
                if element in self.label_flips:
                    y[i] = self.label_flips[element]
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