
# Written by: Oskar
# Last edited: 2022/02/06
# Description: Tests for data module.


# =============================================================================
#  IMPORTS AND DEPENDENCIES
# =============================================================================

import pytest
import numpy as np

from data import DataLoader


# =============================================================================
#  TEST FUNCTIONS
# =============================================================================

def test_shuffler():
    """ Make sure the dataloader can successfully shuffle the data. """
    # Make some dummy data
    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]])
    y = np.array([1, 2, 3, 4, 5])

    # Instantiate a dataloader
    dataloader = DataLoader(X, y, batch_size=5, shuffle=True)

    # Make sure ordering of labels is different
    assert not np.array_equal(dataloader._queue[0][1], y)
    

def test_cache():
    """ Make sure the cache works as expected. """
    # Make some dummy data
    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]])
    y = np.array([1, 2, 3, 4, 5])

    # Instantiate a dataloader
    dataloader = DataLoader(X, y, batch_size=2, shuffle=False)

    # Make sure 1 unbatched point remains in the cache
    assert len(dataloader._cache) == 1

    # Make sure the remaining point is the last one in the provided sequence
    assert np.array_equal(dataloader._cache[0][0], X[-1])
    assert dataloader._cache[0][1] == y[-1]

    # Make some more dummy data, add to the cache
    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]])
    y = np.array([1, 2, 3, 4, 5])
    dataloader.add_to_cache(X, y)

    # Make sure no unbatched points remain in the cache
    assert len(dataloader._cache) == 0


def test_queue():
    """ Make sure the queue works as expected. """
    # Make some dummy data
    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]])
    y = np.array([1, 2, 3, 4, 5])

    # Instantiate a dataloader
    dataloader = DataLoader(X, y, batch_size=2, shuffle=False)

    # Make sure there are 2 batches in the queue
    assert len(dataloader._queue) == 2

    # Make sure the batches are as expected
    assert np.array_equal(dataloader._queue[0][0], X[:2, :])
    assert np.array_equal(dataloader._queue[0][1], y[:2])

    # Make sure the queue empties when iterated over
    for batch in dataloader:
        pass
    assert len(dataloader._queue) == 0
