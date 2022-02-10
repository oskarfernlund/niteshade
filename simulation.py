
# =============================================================================
#  IMPORTS AND DEPENDENCIES
# =============================================================================

import numpy as np
import pickle

from data import DataLoader
from model import IrisClassifier
from copy import deepcopy

#from postprocessing import PostProcessor
from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from utils import save_pickle

class Simulator():
    """"""
    def __init__(self, X, y, model, attacker=None,
                 defender=None, batch_size=1, episode_size=1,
                 save=False, **kwargs):
        """"""
        
        self.X = X
        self.y = y
        self.episode_size = episode_size
        self.batch_size = batch_size
        self.model = model
        self.attacker = attacker
        self.defender = defender
        self.save = save

        self.results = {"X_stream": [], "y_stream": [], "models": []}

    def learn_online(self, verbose=True):
        """"""
        generator = DataLoader(self.X, self.y, batch_size = self.episode_size) #initialise data stream
        batch_queue = DataLoader(batch_size = self.batch_size)
        
        for episode, (X_episode, y_episode) in enumerate(generator):

            # Attacker's turn to attack
            if self.attacker:
                X_episode, y_episode = self.attacker.attack(X_episode, y_episode)

            # Defender's turn to defend
            if self.defender:
                X_episode, y_episode = self.defender.defend(X_episode, y_episode)

            batch_queue.add_to_cache(X_episode, y_episode)
            
            
            # Online learning loop
            for batch_idx, (X_batch, y_batch) in enumerate(batch_queue):

                self.model.step(X_batch, y_batch)

                if verbose:
                    # Print training loss
                    if batch_idx % 10 == 0:
                        print("Train Epoch: {:02d} -- Batch: {:03d} -- Loss: {:.4f}".format(
                            episode,
                            batch_idx,
                            self.model.losses[-1],
                            )
                            )
                
            
            self.results["X_stream"].append(X_episode)
            self.results["y_stream"].append(y_episode)
            self.results["models"].append(deepcopy(self.model.state_dict()))
                
                # Postprocessor saves resultsb
                #postprocessor.cache(databatch, perturbed_databatch, model.epoch_loss)

            # Save the results to the results directory
            if self.save:
                save_pickle(self.results)
                            
