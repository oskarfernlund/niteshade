import numpy as np
import math
from sklearn.utils import shuffle

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
    
    def perturb(self, databatch):
        """Poison a batch of data randomly."""
        for i in range(len(databatch[0])):
            if np.random.randint(2) == 0:
                pass
            else:
                databatch[1][i] = np.array([np.random.randint(3)])
        
        return databatch

            
class SimpleAttacker:
    
    
    def __init__(self, label, aggressiveness, rate = None):
    
        self.label = label
        self.aggressiveness = aggressiveness
        self.rate = rate
        
    def attack(self, x, y):
        """ Adds points to the databatch.
        
        Add a certain number of points (based on the aggressiveness) to 
        the databatch, with the y lable being as the attacker pleases.
        
        Args:
            x (array) : data 
            y : labels of data
            label : label attached to new points added 
        """
        num_to_add = self._num_pts_to_add(x)
        x_add = self._pick_random_data(x, num_to_add)
        x = np.append(x, x_add, axis = 0)
        
        y_add = np.full((num_to_add,1), self.label)
        y = np.append(y, y_add)
        
        x, y = shuffle(x,y)
        
        return x, y
        
            
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
        
            
            
    
if __name__ == "__main__":
    x = np.array([[1,2,3], [1,3,2], [3,4,5],[4,5,6],[3,6,7]])
    y = np.array([1,2,1,3,4])
    np.transpose(y)
    attacker = SimpleAttacker(5, 0.6)
    print(attacker.attack(x,y))
    