
# Written by: Jaime, Mustafa
# Last edited: 2022/04/05
# Description: General utility functions.


# =============================================================================
#  IMPORTS AND DEPENDENCIES
# =============================================================================

import os
import pickle
import numpy as np

from datetime import datetime


# =============================================================================
#  FUNCTIONS
# =============================================================================

def get_time_stamp_as_string():
    """"""
    # Return current timestamp in a saving friendly format.
    date_time = datetime.now()
    date_time_str = date_time.strftime("%d_%b_%Y_(%H_%M_%S.%f)")
    return date_time_str


def save_pickle(results, dir_name='output', file_name='default'):
    """"""
    # Check if dirrectory exists, if not make one
    if not os.path.isdir(dir_name):
        os.mkdir(dir_name)

    # Generate plot name, if needed
    if file_name == 'default':
        file_name = f'{get_time_stamp_as_string()}'

    pickle.dump(results, open(f"{dir_name}/{file_name}", "wb"))


def load_model(filename):
    """Load a binary file containing a neural network.

    Returns:
        model {nn.Module}: Trained model inheriting nn.Module with learnable parameters .
    """
    # If you alter this, make sure it works in tandem with save_regressor
    with open(filename, 'rb') as target:
        model = pickle.load(target)
    
    return model


def save_model(model, filename):
    """Save classifier as a binary .pickle file.

    Args:
            filename {str}: name of file to save model in (excluding extension).
    """
    # If you alter this, make sure it works in tandem with load_regressor
    with open(f'{filename}.pickle', 'wb') as target:
        pickle.dump(model, target)


def one_hot_encoding(y, num_classes):       
    """ Perform one hot encoding of previiously decoded data.
    
    Args:
        y (np.array or torch.tensor) : labels
        num_classes (int) : number of classes 
        
    Returns:
        enc_y (np.array or torch.tensor) : encoded labels
    """
    enc_y = np.zeros([np.shape(y)[0], num_classes])

    for i in range(np.shape(y)[0]):
        element = y[i]
        enc_y[i][int(element)] = 1
        
    return enc_y


def check_num_of_classes(y):
    """Check the number of classes in one hot encoded data.
    
    Supposing data is initially encoded, ti attack it it needs to be decoded.
    Then, before outputting it, it needs to be encoded once again and so we 
    require the number of classes in the data. So we feed in this function the 
    initial encoded labela data to deteremine the number of classes.
    
    Args:
        y (np.array or torch.tensor) : labels (encoded)
     
    Returns:
        num_classes (int) : number of classes
    """
    if check_batch_size(y) == 1:
        y = y.reshape(1,-1)
    num_classes = np.shape(y)[1]
    
    return num_classes
    

def check_batch_size(y):
    """Check the batch size of input label data.
    
    If batch size is 1, we need to reshape data for encoding/decoding.
    The output is not the actual batch size, rather it is checking whether 
    batch size is 1 or not.
    
    Args: 
        y (np.array or torch.tensor) : labels
    
    Returns:
        check (int) : 1 means 1, else means not 1
    """
    check = len(np.shape(y))
    
    return check 
    

def decode_one_hot(y):
    """Decode one hot encoded data.
    
    Args:
        y (np.array or torch.tensor) : labels (encoded)
    
    Returns:
        new_y (np.array or torch.tensor) : labels (decoded)
    """
    if check_batch_size(y) == 1:
        y = y.reshape(1,-1)
          
    num_classes = np.shape(y)[1]
    new_y = np.zeros([np.shape(y)[0], 1])
    for i in range(num_classes):
        y_col_current = y[:,i]
        for j in range(np.shape(y)[0]):
            if y_col_current[j] != 0:
                new_y[j] = i
    return new_y