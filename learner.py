
# =============================================================================
#  IMPORTS AND DEPENDENCIES
# =============================================================================

import numpy as np
import pickle

from datastream import DataStream
from attack import RandomAttacker
from defence import RandomDefender
from model import IrisClassifier
#from postprocessing import PostProcessor
from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from utils import save_pickle

class Learner():
    """"""
    def __init__(self, X, y, model, attacker=None,
                 defender=None, episodes=100, batch_size=1, save=False):
        """"""
        self.datastream = DataStream(X,y, batch_size)
        self.batch_size = batch_size
        self.model = model
        self.attacker = attacker
        self.defender = defender
        self.episodes = episodes

        self.save = save

        self.results = {"X_stream": [], "y_stream": [], "models": []}

        def learn_online(self):
            """"""
            for episodes in range(self.episodes):
                
                # Online learning loop
                while self.datastream.is_online():

                    # Fetch a new datapoint (or batch) from the stream
                    databatch = self.datastream.fetch()

                    # Attacker's turn to perturb
                    if self.attacker:
                        databatch = self.attacker.perturb(databatch)

                    # Defender's turn to defend
                    if self.defender:
                        databatch = self.defender.reject(databatch)

                    X_episode_batch, y_episode_batch = databatch
                    self.model.fit(X_episode_batch, y_episode_batch)

                    self.results["X_stream"].append(X_episode_batch)
                    self.results["y_stream"].append(y_episode_batch)
                    self.results["models"].append(self.model)
                    # Postprocessor saves resultsb
                    #postprocessor.cache(databatch, perturbed_databatch, model.epoch_loss)

                # Save the results to the results directory
                if self.save:
                    save_pickle(self.results)
                            
