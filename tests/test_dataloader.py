#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Unit tests for the dataloader class.
"""


# =============================================================================
#  IMPORTS AND DEPENDENCIES
# =============================================================================

import pytest
import numpy as np
import torch

from niteshade.data import DataLoader


# =============================================================================
#  FUNCTIONS
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


def test_numpy_array_compatibility():
    """ Check the dataloader supports numpy arrays as intended. """
    # Common batch size for all tests because why not
    batch_size = 16

    # Simple basis function regression example
    N, M = 100, 10
    X = np.random.rand(N, M) 
    y = np.random.rand(N)
    dataloader = DataLoader(X, y, batch_size, shuffle=True)
    for batch in dataloader:
        assert batch[0].shape == (batch_size, M)
        assert batch[1].shape == (batch_size,)
        assert type(batch[0]) == np.ndarray
        assert type(batch[1]) == np.ndarray

    # Multilabel basis function regression example
    N, M, L = 100, 10, 3
    X = np.random.rand(N, M) 
    y = np.random.rand(N, L)
    dataloader = DataLoader(X, y, batch_size, shuffle=True)
    for batch in dataloader:
        assert batch[0].shape == (batch_size, M)
        assert batch[1].shape == (batch_size, L)
        assert type(batch[0]) == np.ndarray
        assert type(batch[1]) == np.ndarray

    # Image classification example
    N, C, H, W = 100, 3, 32, 32
    X = np.random.rand(N, C, H, W) 
    y = np.random.rand(N)
    dataloader = DataLoader(X, y, batch_size, shuffle=True)
    for batch in dataloader:
        assert batch[0].shape == (batch_size, C, H, W)
        assert batch[1].shape == (batch_size,)
        assert type(batch[0]) == np.ndarray
        assert type(batch[1]) == np.ndarray

    # Multilabel image classification example
    N, C, H, W, L = 100, 3, 32, 32, 5
    X = np.random.rand(N, C, H, W) 
    y = np.random.rand(N, L)
    dataloader = DataLoader(X, y, batch_size, shuffle=True)
    for batch in dataloader:
        assert batch[0].shape == (batch_size, C, H, W)
        assert batch[1].shape == (batch_size, L)
        assert type(batch[0]) == np.ndarray
        assert type(batch[1]) == np.ndarray


def test_pytorch_tensor_compatibility():
    """ Check the dataloader supports pytorch tensors as intended. """
    # Common batch size for all tests because why not
    batch_size = 16

    # Simple basis function regression example
    N, M = 100, 10
    X = torch.randn(N, M) 
    y = torch.randn(N)
    dataloader = DataLoader(X, y, batch_size, shuffle=True)
    for batch in dataloader:
        assert tuple(batch[0].shape) == (batch_size, M)
        assert tuple(batch[1].shape) == (batch_size,)
        assert type(batch[0]) == torch.Tensor
        assert type(batch[1]) == torch.Tensor

    # Multilabel basis function regression example
    N, M, L = 100, 10, 3
    X = torch.randn(N, M) 
    y = torch.randn(N, L)
    dataloader = DataLoader(X, y, batch_size, shuffle=True)
    for batch in dataloader:
        assert tuple(batch[0].shape) == (batch_size, M)
        assert tuple(batch[1].shape) == (batch_size, L)
        assert type(batch[0]) == torch.Tensor
        assert type(batch[1]) == torch.Tensor

    # Image classification example
    N, C, H, W = 100, 3, 32, 32
    X = torch.randn(N, C, H, W) 
    y = torch.randn(N)
    dataloader = DataLoader(X, y, batch_size, shuffle=True)
    for batch in dataloader:
        assert tuple(batch[0].shape) == (batch_size, C, H, W)
        assert tuple(batch[1].shape) == (batch_size,)
        assert type(batch[0]) == torch.Tensor
        assert type(batch[1]) == torch.Tensor

    # Multilabel image classification example
    N, C, H, W, L = 100, 3, 32, 32, 5
    X = torch.randn(N, C, H, W) 
    y = torch.randn(N, L)
    dataloader = DataLoader(X, y, batch_size, shuffle=True)
    for batch in dataloader:
        assert tuple(batch[0].shape) == (batch_size, C, H, W)
        assert tuple(batch[1].shape) == (batch_size, L)
        assert type(batch[0]) == torch.Tensor
        assert type(batch[1]) == torch.Tensor


# =============================================================================
#  MAIN ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    test_shuffler()
    test_cache()
    test_queue()
    test_numpy_array_compatibility()
    test_pytorch_tensor_compatibility()