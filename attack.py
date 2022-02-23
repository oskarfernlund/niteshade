import numpy as np
import math
from sklearn.utils import shuffle
from sklearn.preprocessing import OneHotEncoder

class Attacker:
    
    
    def __init__(self, aggresiveness):
        pass
        # self.aggresiveness = aggresiveness

    # def _num_pts_to_add(self, x):
        # """ Calculates the number of points to add to the databatch.
        
        # Args:
            # x (array) : data
        # """
        # num_points = len(x)
        # num_to_add = math.floor(num_points * self.aggressiveness)
        # if num_to_add == 0:
            # num_to_add = 1

        # return num_to_add
        
    # def _pick_random_data(self, data, n):
        # """ Pick n random data from x that will be used for attacking points
        
        # Args:
            # x (array) : data
            # n (int) : number of data we will take from x
        # """
        # data = shuffle(data)
        # rows = data[:n,]
        
        # return rows


class AddPointsAttacker(Attacker):
    def __init__(self):
        pass
        

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
    
    def attack(self, X, y):
        """Poison a batch of data randomly."""
        for i in range(len(X)):
            if np.random.randint(2) == 0:
                pass
            else:
                y[i] = np.array([np.random.randint(3)])
        
        return X, y

            
class SimpleAttacker(AddPointsAttacker):
    
    
    def __init__(self, aggressiveness, rate = None):
    
        self.aggressiveness = aggressiveness
        self.rate = rate
        
    def attack(self, x, y, label):
        """ Adds points to the databatch.
        
        Add a certain number of points (based on the aggressiveness) to 
        the databatch, with the y lable being as the attacker pleases.
        
        Args:
            x (array) : data 
            y : labels of data
            label : label attached to new points added 
        """
        num_to_add = super()._num_pts_to_add(x)
        x_add = super()._pick_random_data(x, num_to_add)
        x = np.append(x, x_add, axis = 0)
        
        y_add = np.full((num_to_add,1), label)
        y = np.append(y, y_add)
        
        x, y = shuffle(x,y)
        
        return x, y
                    

        
            
            
    
# if __name__ == "__main__":
    # x = np.array([[1,2,3], [1,3,2], [3,4,5],[4,5,6],[3,6,7]])
    # y = np.array([1,2,1,3,4])
    # np.transpose(y)
    # attacker = SimpleAttacker(0.6)
    # print(attacker.attack(x,y,5))
    

data = np.loadtxt("datasets/iris.dat")

X, y = data[:, :4], data[:, 4:]
# print(y[0])
rand = RandomAttacker()
X_new, y_new = rand.attack(X, y)
print(y[:20], y_new[:20])