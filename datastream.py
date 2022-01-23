
# Written by: Oskar
# Last edited: 2022/01/22
# Description: Data stream module. Contains classes and functions pertaining to
# the delivery of a sequential series of datapoints.


# =============================================================================
#  IMPORTS AND DEPENDENCIES
# =============================================================================

import numpy as np


# =============================================================================
#  CLASSES
# =============================================================================

class DataStream:
    """ Data stream class. 
    
    Stores features (X) and corresponding labels (y) as a sequential stream of
    batches (default batch size is 1). Only one batch of datapoints can be
    retrieved at a time, and the complete stream is private so that it may not
    be accessed outside of the class.
    """

    def __init__(self, X, y, batch_size=1):
        """ Initialise and store the data stream.
        
        Features and labels are batched and paired as tuples in a sequential 
        list, and set as a private attribute so that it may not be accessed 
        outside of the class. Batch size is set as a property so that it is
        accessible but immutable outside of the class.
    
        Args:
            X (np.array) : features (shape = N x D)
            y (np.array) : coresponding labels (shape = N)
            batch_size (int) : size of the batches to generate
        """
        self.__stream = self._create_stream(X, y, batch_size)
        self.__batch_size = batch_size

    def __str__(self):
        """ Represent the class instance as a string. """
        return f"DataStream object with batch size {self.__batch_size}"

    @property
    def batch_size(self):
        """ Set the batch size as a property. """
        return self.__batch_size

    def fetch(self):
        """ Pop the next data batch off the stream queue. """
        return self.__stream.pop(0)

    def is_empty(self):
        """ Check if the stream queue is empty. """
        return not self.__stream

    def _create_stream(self, X, y, batch_size):
        """ Create the stream queue.
        
        Remove datapoints at the end of the stream which cannot fit into a 
        batch (number of points must be evenly divisible by the batch size).
        Batch the features and labels and pair them together as tuples. 

        Args:
            X (np.array) : features (shape = N x D)
            y (np.array) : coresponding labels (shape = N)
            batch_size (int) : size of the batches to generate

        Returns:
            (list of tuple) : sequential batches of (X, y)
        """
        # Make sure X is a 2D array
        if X.ndim == 1:
            X = np.expand_dims(X, axis=1)
        
        # Remove datapoints which do not fit into a batch
        n_points_to_trim = X.shape[0] % batch_size
        if n_points_to_trim > 0:
            X, y = X[:-n_points_to_trim], y[:-n_points_to_trim]

        # Split features and labels into batches
        X_batches = np.split(X, X.shape[0]//batch_size)
        y_batches = np.split(y, y.shape[0]//batch_size)
        batches = zip(X_batches, y_batches)

        return [batch for batch in batches]


# =============================================================================
#  FUNCTIONS
# =============================================================================

def main():
    """ Sample main. """
    # Create some fake data
    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
    y = np.array([1, 2, 3, 4])
    batch_size = 1

    # Create a DataStream instance
    datastream = DataStream(X, y, batch_size)
    print(datastream)
    print(datastream.batch_size)
    
    # Print out the DataStream sequence
    while not datastream.is_empty():
        print(datastream.fetch())
    print("Empty")


# =============================================================================
#  MAIN ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()
