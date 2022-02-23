import numpy as np
import math
from sklearn.utils import shuffle
from sklearn.preprocessing import OneHotEncoder

class Attacker:
    
    
    def __init__(self, aggresiveness, one_hot=False):
        
        self.aggresiveness = aggresiveness
        self.one_hot = one_hot
        
    def check_batch_size(self, y):
        check = len(np.shape(y))

        return check

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
        
    def one_hot_encoding(self, y):       
        encoder = OneHotEncoder()
        encoder.fit(y)
        one_hot = encoder.transform(y).toarray()
        return one_hot
        
        


class AddPointsAttacker(Attacker):
    def __init__(self):
        super().__init__(self, one_hot)
        
        

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
        y = y[0]
        orignal_y = y
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
                y = super().one_hot_encoding(y)
        
        return x, y
                    

        
            
            
    
if __name__ == "__main__":
    # x = np.array([[1,2,3], [1,3,2], [3,4,5],[4,5,6],[3,6,7]])
    # y = np.array([1,2,1,3,4])
    data = np.loadtxt("datasets/iris.dat")

    x, y = data[:, :4], data[:, 4:]
    x = x[0]
    y = y[0]


    attacker = SimpleAttacker(0.6, 1, one_hot=True)
    print(attacker.attack(x,y)) 
    

