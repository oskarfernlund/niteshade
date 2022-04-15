#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Classes and functions pertaining to the loading and batching of data.
"""


# =============================================================================
#  IMPORTS AND DEPENDENCIES
# =============================================================================

import numpy as np
import torch


# =============================================================================
#  CLASSES
# =============================================================================

class DataLoader:
    """ DataLoader class. 

    Contains a cache and a queue. Features (X) and labels (y) can be added to 
    the cache either by passing them as inputs in the constructor, or by 
    calling the method add_to_cache(X, y). 
    
    When data is added to the cache, the points are automatically batched into
    arrays of length batch_size, removed from the cache and added to the queue.
    For example, if a dataset containing 10 points is added to the cache with a
    batch size of 3, 3 batches of length 3 will be created, removed from the 
    cache and added to the queue. In the end, the queue will contain 3 batches
    of size 3 and the cache will contain a single datapoint. If more (X, y) 
    values are added to the cache by calling add_to_cache(X, y), they will be 
    appended to the cache and the batching/queuing process will repeat, this 
    time with the single datapoint at the front of the cache (first in line to 
    be batched and added to the queue).

    The class is an iterator, and returns batches from the queue when iterated
    over until the queue is empty. If more (X, y) values get added to the 
    cache, the class instance may be iterated over again and new batches will
    be produced. If the shuffle argument is set to True, any data added to the
    cache will be shuffled prior to batching.
    """
    def __init__(self, X=None, y=None, batch_size=1, shuffle=False, seed=69):
        """ Initialise the DataLoader.
        
        Features (X) and labels (y) may be passed as inputs in the constructor,
        but this is not necessary. If they are set to their default values of
        None, the cache and queue will initially be empty.
    
        Args:
            X (np.ndarray or torch.tensor) : features (first dimension = N)
            y (np.ndarray or torch.tensor) : labels (first dimension = N)
            batch_size (int) : size of the batches to generate
            shuffle (bool) : whether or not to shuffle the datapoints before 
            seed (int) : seed for the random number generator 
        """
        # Set/initialise attributes
        self.batch_size = batch_size
        self.shuffle = shuffle
        self._cache = []
        self._queue = []
        
        # Initialise the random number generator (for shuffling)
        if shuffle:
            self._rng = np.random.default_rng(seed)
        else:
            self._rng = None
        
        # If data has been passed as an argument, add it to cache/queue
        if X is not None and y is not None:
            self.add_to_cache(X, y)

    def __iter__(self):
        """ Return the iterator object (implicitly called before loops). """
        return self

    def __next__(self):
        """ Return the next value in the sequence.
        
        Implicitly called at each loop increment. Raises a StopIteration 
        exception when there are no more values to return (queue is empty),
        which is implicitly captured by looping constructs to stop iterating.
        """
        try:
            X, y = self._queue.pop(0)
        except IndexError:
            raise StopIteration
        return X, y

    def __str__(self):
        """ Represent the class instance as a string. """
        return f"DataLoader object with batch size {self.batch_size}"

    def __len__(self):
        """ Returns the length of the queue. """
        return len(self._queue)

    def add_to_cache(self, X, y):
        """ Add features (X) and labels (y) to the cache.

        If shuffle was set to true in the constructor, the datapoints are 
        shuffled before being added to the cache. After points are added to the
        cache, the _cache_to_queue() method is called automatically, which 
        creates as many batches as possible based on the batch size, removes 
        them from the cache and adds them to the queue.

        Args:
            X (np.ndarray or torch.tensor) : features (first dimension = N)
            y (np.ndarray or torch.tensor) : labels (first dimension = N)
        """
        # Make sure X and y dimensions are compatible
        assert X.shape[0] == y.shape[0], "First dim. of X & y must be aligned!"

        # Shuffle the datapoints if shuffle was set to true in the constructor
        if self.shuffle:
            shuffler = self._rng.permutation(X.shape[0])
            X, y = X[shuffler], y[shuffler]

        # Append each datapoint to the cache
        for datapoint in zip(X, y):
            self._cache.append(datapoint)

        # Batch the datapoints; clear them from cache; add them to queue
        self._cache_to_queue()
        
    def _cache_to_queue(self):
        """ Batch cached datapoints, clear them from cache, add them to queue.

        If the number of points in the cache isn't divisible by the batch size, 
        some points will remain in the cache. These will be the first points to
        get batched if new (X, y) values are added to the cache.

        Raises:
            TypeError : if the features and labels (X, y) are neither of type
                np.ndarray nor torch.tensor
        """
        # Repeat until there are insufficient points to form a batch
        while len(self._cache) >= self.batch_size:

            # Extract a batch from the cache; clear it from the cache
            batch = self._cache[:self.batch_size]
            del self._cache[:self.batch_size]

            # Stack datapoints into batch of correct dimensions
            if type(batch[0][0]) == np.ndarray:
                X = np.stack([datapoint[0] for datapoint in batch], axis=0)
                y = np.stack([datapoint[1] for datapoint in batch], axis=0)
            elif type(batch[0][0] == torch.Tensor):
                X = torch.stack([datapoint[0] for datapoint in batch], axis=0)
                y = torch.stack([datapoint[1] for datapoint in batch], axis=0)
            else:
                raise TypeError("Data must be np.ndarray or torch.tensor")

            # Add batch to the queue
            self._queue.append((X, y))


# =============================================================================
#  MAIN ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    pass