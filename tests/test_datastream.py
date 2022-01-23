
# Written by: Oskar
# Last edited: 2022/01/22
# Description: Data stream test module.


# =============================================================================
#  IMPORTS AND DEPENDENCIES
# =============================================================================

import pytest
import numpy as np

from datastream import DataStream


# =============================================================================
#  TEST FUNCTIONS
# =============================================================================

def test_datastream_ends():
    """ Make sure the data stream's fetch method works. """
    # Create some dummy data
    X = np.array([[1, 2, 3], [4, 5, 6]])
    y = np.array([1, 2])
    batch_size = 1

    # Call the fetch method until the data stream depletes
    datastream = DataStream(X, y, batch_size)
    datastream.fetch()
    datastream.fetch()

    assert not datastream.is_online()


def test_datastream_batches():
    """ Make sure the data stream discards points which can't be batched. """
    # Create some dummy data
    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]])
    y = np.array([1, 2, 3, 4, 5])
    batch_size = 2

    # Create datastream
    datastream = DataStream(X, y, batch_size)

    # Count the batches
    batch_count = 0
    while datastream.is_online():
        datastream.fetch()
        batch_count += 1

    assert batch_count == 2


def test_datastream_dimensions():
    """ Make sure the data stream dimensions are correct. """
    # Create some dummy data
    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]])
    y = np.array([1, 2, 3, 4, 5])
    batch_size = 2

    # Create datastream, fetch batch
    datastream = DataStream(X, y, batch_size)
    X_batch, y_batch = datastream.fetch()

    assert X_batch.shape[0] == 2
    assert X_batch.shape[1] == 3
    assert y_batch.shape[0] ==2

    
