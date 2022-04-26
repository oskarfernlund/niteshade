#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Unit tests for the attacker classes.
"""


# =============================================================================
#  IMPORTS AND DEPENDENCIES
# =============================================================================

import pytest
import numpy as np
import torch

from niteshade.attack import LabelFlipperAttacker, AddLabeledPointsAttacker
from niteshade.attack import RandomAttacker, BrewPoison, ChangeLabelAttacker


# =============================================================================
#  FUNCTIONS
# =============================================================================

def test_LabelFlipperAttacker():
    dict = {1:2, 3:4}
    attacker = LabelFlipperAttacker(1, dict)
    X = np.random.rand(6,3)
    y = np.array([1,0,0,3,0,0])
    new_X, new_y = attacker.attack(X, y)
    
    # test correct operation for a simple case
    assert np.array_equal(new_y, np.array([2,0,0,4,0,0]))
    
    TX = torch.tensor(X)
    Ty = torch.tensor(y)
    new_TX, new_Ty = attacker.attack(TX, Ty)
    
    # test output typs is tensor when input type is tensor
    assert [type(new_TX), type(new_Ty)] == [torch.Tensor,torch.Tensor]
    
    one_hot_y = np.array([[1,0,0],[0,1,0],[0,0,1],[1,0,0],[0,1,0],[0,0,1]])
    dict_2 = {0:1}
    attacker_2 = LabelFlipperAttacker(1, dict_2, True)
    new_one_X, new_one_y = attacker_2.attack(X, one_hot_y)
    
    # test correct operation with one_hot data
    expected = np.array([[0,1,0],[0,1,0],[0,0,1],[0,1,0],[0,1,0],[0,0,1]])
    assert np.array_equal(new_one_y, expected)
    
def test_AddLabeledPointsAttacker():
    attacker = AddLabeledPointsAttacker(1, 0)
    X = np.random.rand(10, 3)
    y = np.random.randint(0, 3, size=10)
    new_X, new_y = attacker.attack(X, y)
    
    # check max added number of points <= existing number of points
    assert len(new_X) <= 20
    
    # ensure shuffling after addition
    assert (np.array_equal(X, new_X[10]) == False)

    # ensure number of added points corresponds to number of added labels
    assert len(new_X) == len(new_y)
    
    TX = torch.tensor(X)
    Ty = torch.tensor(y)
    new_TX, new_Ty = attacker.attack(TX, Ty)
    
    # test output typs is tensor when input type is tensor
    assert [type(new_TX), type(new_Ty)] == [torch.Tensor,torch.Tensor]
    
def test_RandomAttacker():
    attacker = RandomAttacker(0.5)
    X = np.random.rand(10,3)
    y = np.array([0,1,2,0,1,2,0,1,2,0])
    old = y
    new_X, new_y = attacker.attack(X, y)
    super = ChangeLabelAttacker(0.5)
    num_to_change = super.num_pts_to_change(X)

    # check num_pts_to_change method in super class
    assert num_to_change == 5

    # check X is unchanged and y is changed
    og = np.array([0,1,2,0,1,2,0,1,2,0])
    assert np.array_equal(X, new_X)
    assert (np.array_equal(y, og) == False)
    
    TX = torch.tensor(X)
    Ty = torch.tensor(y)
    new_TX, new_Ty = attacker.attack(TX, Ty)
    
    # test output typs is tensor when input type is tensor
    assert [type(new_TX), type(new_Ty)] == [torch.Tensor,torch.Tensor]
    
    one_hot_X = np.random.rand(6,3)
    one_hot_y = np.array([[1,0,0],[0,1,0],[0,0,1],[1,0,0],[0,1,0],[0,0,1]])
    attacker_2 = RandomAttacker(0.5, True)
    new_one_X, new_one_y = attacker_2.attack(one_hot_X, one_hot_y)
    
    # test correct operation with one_hot data
    assert np.shape(new_one_y)[1] == 3
    
def test_BrewPoison():
    attacker = BrewPoison(0)
    X_1 = torch.tensor([[1,2],[3,4]])
    X_2 = torch.tensor([[1,3],[2,4]])
    X_3 = torch.tensor([[1,4],[2,3]])
    X_list = [X_1, X_2, X_3]
    pert = torch.tensor([[1,1],[1,1]])
    pert_list = attacker.apply_pert(X_list, pert)
    
    # test apply_pert method
    pert_1 = torch.tensor([[2,3],[4,5]])
    pert_2 = torch.tensor([[2,4],[3,5]])
    pert_3 = torch.tensor([[2,5],[3,4]])
    exp_list = [pert_1, pert_2, pert_3]
    for i in range(len(exp_list)):
        assert torch.equal(exp_list[i], pert_list[i])
        
    X = torch.rand(3, 28, 28)
    pert = torch.rand(3, 28, 28)
    alpha = 0.5
    new_pert = attacker.get_new_pert(pert, alpha, X)
    
    # test get_new_pert method
    inf_norm = torch.max(torch.abs(pert.reshape(-1,1)))
    limit = 0.5*inf_norm
    inf_norm_new = torch.max(torch.abs(new_pert.reshape(-1,1)))
    assert inf_norm_new <= limit
    
    curr = 10
    total = 20
    new_curr = attacker.inc_reset_ep(curr, total)
    
    # test inc ep 
    assert new_curr == 11
    
    curr = 19
    new_curr = attacker.inc_reset_ep(curr, total)
    
    # test reset ep
    assert new_curr == 0
    
    
# =============================================================================
#  MAIN ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    pass