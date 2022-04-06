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
from model import IrisClassifier
from tqdm import tqdm
import numpy as np

import matplotlib.pyplot as plt 


class PostProcessor:
    def __init__(self, wrap_models, batch_size, episode_size,
    base_model, epochs=100):
        
        self.wrap_models = wrap_models
        self.batch_size = batch_size
        self.episode_size = episode_size
        self.base_model = base_model
        self.epochs = epochs
        
    # =========================================================================
    #  Utils
    # =========================================================================
    
    def get_time_stamp_as_string(self):
        # Return current timestamp in a saving friendly format.
        date_time = datetime.now()
        date_time_str = date_time.strftime("%d_%b_%Y_(%H_%M_%S.%f)")
        return date_time_str


    def save_plot(self, plt, save_dir='output', plot_name='default'):
        # Check if dirrectory exists, if not make one
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)

        # Generate plot name, if needed
        if plot_name == 'default':
            plot_name = f'{self.get_time_stamp_as_string()}.png'

        plt.savefig(f'{save_dir}/{plot_name}')


    # =========================================================================
    #  Analytics Tools
    # =========================================================================
    
    def compute_accuracies(self, X_test, y_test):
        """
        # Returns a dictionary of lists with accuracies
        """
        simulator_keys = self.wrap_models.keys()
        accuracies = {}
        
        for model_name, list_of_models in tqdm(self.wrap_models.items()):
            for model_specs in list_of_models:
                model = self.base_model
                model.load_state_dict(model_specs)
                _, test_accuracy = model.evaluate(X_test, y_test, self.batch_size)
                if model_name in accuracies:
                    accuracies[model_name].append(test_accuracy)
                else:
                    accuracies[model_name] = [test_accuracy]
        return accuracies

    def plot_online_learning_accuracies(self, X_test, y_test, save=True):
        """
        # Prints a plot into a console
        """
        accuracies = self.compute_accuracies(X_test, y_test)
        
        x = [e for e in range(len(accuracies['regular']))]

        fig, ax = plt.subplots(1, figsize=(15,10))
        for model_name, accuracy in accuracies.items():
            ax.plot(x, accuracy, label=model_name)
            ax.legend()

        ax.set_title('Test Accuracy Online Learing')
        ax.set_xlabel('Episodes')
        ax.set_ylabel('Accuracy')
        plt.show()
        
        if save: 
            self.save_plot(plt)