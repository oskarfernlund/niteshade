from sklearn import datasets
from sklearn.utils import shuffle

import numpy as np



iris = datasets.load_iris()
X = np.array(iris.data)
y = np.array(iris.target)
print(X,y)
X, y = shuffle(X, y)
print(X,y)

# class Attacker:
    # """ Attacks the current datapoint.
    
    # Reads the current data point from the fetch method of DataStream
    # and decides whether or not to attack it. Decision to attack 
    # depends on user, as does the method of attack. 
    # """
    # def __init__(self, databatch):
        # self.databatch = databatch
        # self.batch_size = len(databatch)
    
    # def random_attacker(self, databatch):
        # for i in range(self.batch_size):
            # if np.random.randint(2) == 0:
                # pass
            # else:
                
            
                