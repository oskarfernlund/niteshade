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
import torchvision.transforms as transforms
import torch
from sklearn.utils import shuffle
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from matplotlib.colors import ListedColormap


# =============================================================================
#  FUNCTIONS
# =============================================================================

def copy(array_like):
    """Used to copy a torch.Tensor or np.ndarray object.

    Args:
        array_like (np.ndarray, torch.Tensor) : array/tensor to copy.
    """
    if isinstance(array_like, torch.Tensor):
        return array_like.detach().clone()
    elif isinstance(array_like, np.ndarray):
        return array_like.copy()
    else: 
        raise TypeError("Niteshade only supports np.ndarray or torch.Tensor array-like objects.")

def get_time_stamp_as_string():
    """Get the current time stamp as a string.
    
    Returns:
        date_time_str (str) : current timestemp
    """
    # Return current timestamp in a saving friendly format.
    date_time = datetime.now()
    date_time_str = date_time.strftime("%d-%b-%Y (%H-%M-%S)")
    return date_time_str


def save_plot(plt, dirname='output', plotname='default'):
    """Save a matplotlib.pyplot.plot() as a .png file.
    
    Args:
        dirname (str) : name of the directory to save the plot to. 
                        If the directory doesn't exist, it is created
        plotname (str): plot name, if plotname is set to default, 
                        the plot name is set to timestemp
    """
    # Check if dirrectory exists, if not make one
    if not os.path.isdir(dirname):
        os.mkdir(dirname)

    # Generate plot name, if needed
    if plotname == 'default':
        plotname = f'{get_time_stamp_as_string()}.png'

    plt.savefig(f'{dirname}/{plotname}')


def save_pickle(results, dirname='output', filename='default'):
    """Save results as a .pickle file.
    
    Args:
        dirname (str) : name of the directory to save the pickle to. 
                        If the directory doesn't exist, it is created
        filename (str) : file name, if filename is set to default, 
                        the file name is set to timestemp
    """
    # Check if dirrectory exists, if not make one
    if not os.path.isdir(dirname):
        os.mkdir(dirname)

    # Generate plot name, if needed
    if filename == 'default':
        filename = f'{get_time_stamp_as_string()}'

    pickle.dump(results, open(f"{dirname}/{filename}", "wb"))


def load_model(filename):
    """Load a binary file containing a neural network.

    Args: 
        filename (str) : name of file to save model in (excluding extension)

    Returns:
        model (nn.Module) : Trained model inheriting nn.Module with learnable parameters
    """
    # If you alter this, make sure it works in tandem with save_regressor
    with open(filename, 'rb') as target:
        model = pickle.load(target)
    
    return model


def save_model(model, filename):
    """Save an object as a binary .pickle file.

    Args:
        model (nn.Module) : model to save
        filename (str) : name of file to save model in (excluding extension)
    """
    # If you alter this, make sure it works in tandem with load_regressor
    with open(f'{filename}.pickle', 'wb') as target:
        pickle.dump(model, target)


def one_hot_encoding(y, num_classes):       
    """ Perform one hot encoding of previiously decoded data.
    
    Args:
        y (np.array, torch.tensor) : labels
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
    
    Supposing data is initially encoded, to attack it needs to be decoded.
    Then, before outputting it, it needs to be encoded once again and so we 
    require the number of classes in the data. So we feed in this function the 
    initial encoded labela data to deteremine the number of classes.
    
    Args:
        y (np.array, torch.tensor) : labels (encoded)
     
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
        y (np.array, torch.tensor) : labels
    
    Returns:
        check (int) : 1 means 1, else means not 1
    """
    check = len(np.shape(y))
    
    return check 
    

def decode_one_hot(y):
    """Decode one hot encoded data.
    
    Args:
        y (np.array, torch.tensor) : labels (encoded)
    
    Returns:
        new_y (np.array, torch.tensor) : labels (decoded)
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


def train_test_iris(test_size=0.2, val_size = None, rand_state=42):
    """Loads the Iris dataset using sklearn.datasets.load_iris()
    and returns the inputs and labels in splitted train and test
    sets (and validation too if val_size != None). Please note 
    that the returned labels are one-hot encoded.

    Args: 
        test_size (float) : Size of the test set expressed as a fraction
                            of the complete dataset (Default = 0.2)
        val_size (float) : Size of the validation set expressed as a fraction
                           of the training set. Default = None (i.e only train 
                           and test sets are returned)
        rand_state (int) : random seed with which to split the dataset using
                           sklearn.model_selection.train_test_split(). 
                           Default = 42
    
    Returns: 
        X_train (np.ndarray) : Train inputs
        y_train (np.ndarray) : Train labels
        X_test (np.ndarray) : Test inputs
        y_test (np.ndarray) : Test labels     
        X_val (np.ndarray) : Validation inputs (only if val_size != None)
        y_val (np.ndarray) : Validation labels (only if val_size != None)   
    """
    #define input and target data
    data = load_iris()

    #define input and target data
    X, y = data.data, data.target

    #one-hot encode
    enc = OneHotEncoder()
    y = enc.fit_transform(y.reshape(-1,1)).toarray()

    #split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, 
                                                        random_state=rand_state)

    #normalise data using sklearn module
    scaler = MinMaxScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    X_train, y_train = shuffle(X_train, y_train)

    return X_train, y_train, X_test, y_test


def train_test_MNIST(dir="datasets/", transform=None, val_size=None):
    """Function to load torchivisions' MNIST dataset, splitted into 
    train, test, and validation sets (the latter only if val_size != None).

    Args:
        transform (torchvision.transforms) : Sequence of transformations to apply
                                             to the train and test sets.
                                             Default: transforms.Compose([torchvision.transforms.Normalize(
                                                                          (0.1307,), (0.3081,)), 
                                                                          transforms.ToTensor()]))
        val_size (float) : Value between 0 and 1 indicating the percentage of the
                           training set that should be allocated to the validation set. 
                           (Default = 0.2)
    Returns: 
        X_train (np.ndarray) : Train inputs
        y_train (np.ndarray) : Train labels
        X_test (np.ndarray) : Test inputs
        y_test (np.ndarray) : Test labels    
        X_val (np.ndarray) : Validation inputs (only if val_size != None)
        y_val (np.ndarray) : Validation labels (only if val_size != None) 
    """
    if transform is None:
        transform = torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))])

    MNIST_train = torchvision.datasets.MNIST(root=dir, train=True, download=True,
                                             transform=transform)

    #get inputs and labels and convert to numpy arrays
    X = MNIST_train.data.numpy().reshape(-1, 1, 28, 28)
    y = MNIST_train.targets.numpy()

    MNIST_test = torchvision.datasets.MNIST(root=dir, train=False, download=True,
                                            transform=transform)
    
    X_test = MNIST_test.data.numpy().reshape(-1, 1, 28, 28)
    y_test = MNIST_test.targets.numpy()

    if val_size:
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=val_size) 
        return X_train, X_val, X_test, y_train, y_val, y_test
    else:
        return X, y, X_test, y_test


def train_test_cifar(dir="datasets/", transform = None, val_size=None):
    """
    Function to load torchvisions' CIFAR10 dataset, splitted into 
    train, test, and validation sets (the latter only if val_size != None).

    Args:
        transform (torchvision.transforms) : Sequence of transformations to apply
                                             to the train and test sets.
                                             Default: transforms.Compose([transforms.RandomHorizontalFlip(), 
                                                                          transforms.ToTensor()]))
        val_size (float) : Value between 0 and 1 indicating the percentage of the
                           training set that should be allocated to the validation set. 
                           (Default = 0.2)
    Returns: 
        X_train (torch.Tensor) : Train inputs
        y_train (torch.Tensor) : Train labels
        X_test (torch.Tensor) : Test inputs
        y_test (torch.Tensor) : Test labels      
        X_val (torch.Tensor) : Validation inputs (only if val_size != None)
        y_val (torch.Tensor) : Validation labels (only if val_size != None)
    """
    if transform is None:    
        #data augmentation transformations for train and val/test datasets
        transform = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor()])

    train = torchvision.datasets.CIFAR10(root=dir, train=True, download=True, transform=transform)
    test = torchvision.datasets.CIFAR10(root=dir, train=False, download=True, transform=transform)

    X = torch.stack([sample[0] for sample in train]).reshape(-1,3,32,32)
    y = torch.stack([torch.tensor(sample[1]) for sample in train])

    X_test = torch.stack([sample[0] for sample in test]).reshape(-1,3,32,32)
    y_test = torch.stack([torch.tensor(sample[1]) for sample in test])

    if val_size:
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=val_size) 
        return X_train, y_train, X_test, y_test, X_val, y_val
    else:
        return X, y, X_test, y_test


def rand_cmap(nlabels):
    """
    Creates a random colormap to be used together with matplotlib. Useful for segmentation tasks.

    Args:
        nlabels (int) : Number of labels (size of colormap)

    Returns:
         random_colormap (ListedColormap) : colormap filled with nlabels random colors
    """

    # Generate color map for bright colors, based on hsv
    randHSVcolors = [(np.random.uniform(low=0.0, high=1),
                        np.random.uniform(low=0.2, high=1),
                        np.random.uniform(low=0.9, high=1)) for _ in range(nlabels)]

    # Convert HSV list to RGB
    randRGBcolors = []
    for HSVcolor in randHSVcolors:
        randRGBcolors.append(colorsys.hsv_to_rgb(HSVcolor[0], HSVcolor[1], HSVcolor[2]))

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
