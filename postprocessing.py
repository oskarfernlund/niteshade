# Written by: Alexandra
# Last edited: 2022/01/25
# Description: Postprocessing pipeline. This contains analytics tools that 
# assess the damage of the attacker / impact of the defender.


# =============================================================================
#  IMPORTS AND DEPENDENCIES
# =============================================================================

import os
from datetime import datetime
import pickle

import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns


class PostProcessor:
    def __init__(self, data_filename):
        
        # Load data: each epoch contains batches of the following tuples
        # [(X, y, X_poisoned, y_poisoned, X_filtered, y_filtered,
        # loss_X, loss_x_poisoned, loss_X_filtered)]
        
        with open(data_filename, 'rb') as data:
            cross_epoch_data = pickle.load(data)
        
        self.cross_epoch_data = np.ndarray(cross_epoch_data)


    # =========================================================================
    #  Utils
    # =========================================================================
    
    def get_time_stamp_as_string(self):
        # Return current timestamp in a saving friendly format.
        date_time = datetime.now()
        date_time_str = date_time.strftime("%d_%b_%Y_(%H_%M_%S.%f)")
        return date_time_str


    def save_plot(self, fig, save_dir='output', plot_name='default'):
        # Check if dirrectory exists, if not make one
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)

        # Generate plot name, if needed
        if plot_name == 'default':
            plot_name = f'{self.get_time_stamp_as_string()}.png'

        plt.savefig(f'{save_dir}/{plot_name})


    # =========================================================================
    #  Analytics Tools
    # =========================================================================
    
    def loss_summary(self):
        """
        Returns a 2D numpy array containing average losses across a batch,
        that correspond to loss_X, loss_x_poisoned, loss_X_filtered.
        The size of the array is equal to the number of epochs.
        """
        
        # Extracting loss columns only
        # TODO: check dimensions when data is available
        cross_epoch_loss = self.cross_epoch_data[:,-3:]
        epoch_loss_averages = cross_epoch_loss.mean(axis = 1)
        return epoch_loss_averages


    def plot_loss_summary(self, save_dir='output', plot_name='default'):
        """
        Plots the loss summary across epochs.
        Please enter save_dir in a string format.
        If the directory doesn't exist, a folder will be created 
        in the working directory.
        """

        fig = plt.figure(figsize=(7,7))
        epoch_loss_averages = self.loss_summary()
        
        # TODO: add legend, formatting at a later stage
        plt.plot(epoch_loss_averages)
        self.save_plot(fig, save_dir=save_dir, plot_name=plot_name)
