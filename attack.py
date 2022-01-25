import numpy as np

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

            
                
