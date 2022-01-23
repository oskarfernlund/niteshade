import numpy as np

class RandomAttacker:
    """ Attacks the current datapoint.
    
    Reads the current data point from the fetch method of DataStream
    and decides whether or not to attack it. Decision to attack 
    depends on user, as does the method of attack. 
    """
    def __init__(self, databatch):
        """
        
        Args:
            databatch (tuple) : batch of data from DataStream.fetch
        """
        
        self.databatch = databatch
        self.batch_size = len(databatch[0])
    
    def perturb(self, databatch):
        for i in range(self.batch_size):
            if np.random.randint(2) == 0:
                pass
            else:
                databatch[1][i] = np.array([np.random.randint(3)])
        
        return databatch

            
                