
# =============================================================================
#  IMPORTS AND DEPENDENCIES
# =============================================================================

import numpy as np
import pickle

from data import DataLoader
from attack import RandomAttacker
from defence import RandomDefender
from model import IrisClassifier
#from postprocessing import PostProcessor
from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from utils import save_pickle

class Simulator():
    """"""
    def __init__(self, X, y, model, attacker=None,
                 defender=None, episodes=100, batch_size=1, save=False):
        """"""
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.model = model
        self.attacker = attacker
        self.defender = defender
        self.episodes = episodes
        self.save = save

        self.results = {"X_stream": [], "y_stream": [], "models": []}

    def learn_online(self, verbose=True):
        """"""
        for episode in range(self.episodes):

            loader = DataLoader(self.X, self.y, self.batch_size) #initialise data stream
            
            # Online learning loop
            for batch_idx, databatch in enumerate(loader):

                # Attacker's turn to perturb
                if self.attacker:
                    databatch = self.attacker.perturb(databatch)

                # Defender's turn to defend
                if self.defender:
                    if self.defender.rejects(databatch):
                        continue

                X_episode_batch, y_episode_batch = databatch
                self.model.step(X_episode_batch, y_episode_batch)

                if verbose:
                    # Print training loss
                    if batch_idx % 10 == 0:
                        print("Train Epoch: {:02d} -- Batch: {:03d} -- Loss: {:.4f}".format(
                            episode,
                            batch_idx,
                            self.model.losses[-1],
                            )
                            )

                self.results["X_stream"].append(X_episode_batch)
                self.results["y_stream"].append(y_episode_batch)
                self.results["models"].append(self.model)
                
                # Postprocessor saves resultsb
                #postprocessor.cache(databatch, perturbed_databatch, model.epoch_loss)

            # Save the results to the results directory
            if self.save:
                save_pickle(self.results)
                            
