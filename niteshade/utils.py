#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
General utility and helper functions.
"""


# =============================================================================
#  IMPORTS AND DEPENDENCIES
# =============================================================================

import os
import pickle
import random
import colorsys
from datetime import datetime

import numpy as np
import torchvision
from sklearn.utils import shuffle
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from matplotlib.colors import ListedColormap


# =============================================================================
#  FUNCTIONS
# =============================================================================

def get_time_stamp_as_string():
    """"""
    # Return current timestamp in a saving friendly format.
    date_time = datetime.now()
    date_time_str = date_time.strftime("%d_%b_%Y_(%H_%M_%S.%f)")
    return date_time_str


def save_plot(plt, save_dir='output', plot_name='default'):
    # Check if dirrectory exists, if not make one
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    # Generate plot name, if needed
    if plot_name == 'default':
        plot_name = f'{get_time_stamp_as_string()}.png'

    plt.savefig(f'{save_dir}/{plot_name}')


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


def train_test_iris(num_stacks = 10):
    #define input and target data
    data = load_iris()

    #define input and target data
    X, y = data.data, data.target

    #one-hot encode
    enc = OneHotEncoder()
    y = enc.fit_transform(y.reshape(-1,1)).toarray()

    #stack data
    X = np.repeat(X, num_stacks, axis=0)
    y = np.repeat(y, num_stacks, axis=0)

    #split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    #normalise data using sklearn module
    scaler = MinMaxScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    X_train, y_train = shuffle(X_train, y_train)

    return X_train, y_train, X_test, y_test


def train_test_MNIST():
    MNIST_train = torchvision.datasets.MNIST('datasets/', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ]))

    #get inputs and labels and convert to numpy arrays
    X_train = MNIST_train.data.numpy().reshape(-1, 1, 28, 28)
    y_train = MNIST_train.targets.numpy()

    MNIST_test = torchvision.datasets.MNIST('datasets/', train=False, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ]))
    
    X_test = MNIST_test.data.numpy().reshape(-1, 1, 28, 28)
    y_test = MNIST_test.targets.numpy()

    return X_train, y_train, X_test, y_test


def rand_cmap(nlabels, type='bright', first_color_black=True, last_color_black=False, verbose=True):
    """
    Creates a random colormap to be used together with matplotlib. Useful for segmentation tasks
    :param nlabels: Number of labels (size of colormap)
    :param type: 'bright' for strong colors, 'soft' for pastel colors
    :param first_color_black: Option to use first color as black, True or False
    :param last_color_black: Option to use last color as black, True or False
    :param verbose: Prints the number of labels and shows the colormap. True or False
    :return: colormap for matplotlib
    """
    if type not in ('bright', 'soft'):
        print ('Please choose "bright" or "soft" for type')
        return

    # Generate color map for bright colors, based on hsv
    if type == 'bright':
        randHSVcolors = [(np.random.uniform(low=0.0, high=1),
                          np.random.uniform(low=0.2, high=1),
                          np.random.uniform(low=0.9, high=1)) for i in range(nlabels)]

        # Convert HSV list to RGB
        randRGBcolors = []
        for HSVcolor in randHSVcolors:
            randRGBcolors.append(colorsys.hsv_to_rgb(HSVcolor[0], HSVcolor[1], HSVcolor[2]))

        if first_color_black:
            randRGBcolors[0] = [0, 0, 0]

        if last_color_black:
            randRGBcolors[-1] = [0, 0, 0]

        random_colormap = ListedColormap(randRGBcolors, N=nlabels)

    # Generate soft pastel colors, by limiting the RGB spectrum
    if type == 'soft':
        low = 0.6
        high = 0.95
        randRGBcolors = [(np.random.uniform(low=low, high=high),
                          np.random.uniform(low=low, high=high),
                          np.random.uniform(low=low, high=high)) for i in range(nlabels)]

        if first_color_black:
            randRGBcolors[0] = [0, 0, 0]

        random_colormap = ListedColormap(randRGBcolors, N=nlabels)

    return random_colormap


def get_cmap(nlabels):
    rgb_distinct =  [(0.588,0.294,0), #brown
                     (1,0.647,0), #orange
                     (0.502,0.502,0), #olive
                     (0,0.53,0.55), #green
                     (0,1,1), #cyan
                     (0,0,0.93), #blue
                     (1,0,0), #red
                     (1,0.412,0.706), #pink
                     (1,1,0), #yellow
                     (0,0,0), 
                     (0.627,0.125,0.941)] #black
    
    try: 
        colors = random.sample(rgb_distinct, nlabels)
        cmap = ListedColormap(colors, N=nlabels)
    except IndexError: 
        cmap = rand_cmap(nlabels)
    
    return cmap


# =============================================================================
#  MAIN ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    pass
