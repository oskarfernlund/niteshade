
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

def _get_simulators(simulators, exclude):
    """Utility function to get simulators given user exclusion requirements.
    """
    if exclude is None:
        return simulators

    else:
        simulators = {}
        for sim_type, simulator in simulators.items():
            if sim_type not in exclude:
                simulators[sim_type] = simulator

        return simulators


def run_simulations(X, y, model, attacker,
                    defender, batch_size=1, episode_size=1,
                    save=False, verbose = True, exclude = None):
    """Wrap results for all possible simulations that can be run
    given the attack and defence strategies provided.

    Args:
            exclude {list}: combinations to exclude from wrapper; Default = None.

            Possible elements of list: 'only_attack' = simulation that dismisses defense strategy.
            
                                       'only_defense' = simulation that dismisses attack strategy.
                                    
                                       'attack_and_defence' = simulation that includes both strategies.
                                    
                                       'regular' = simulation that trains model regularly without
                                                   attack and defense strategies.

    Returns:
            all_results_X {dict}: Input data for the models of each simulation.

            all_results_y {dict}: Target data for the models of each simulation.

            all_models {dict}: Models for each simulation (models of type nn.Module).
    """

    regular = Simulator(X, y, model, attacker=None, defender=None,
                                batch_size=batch_size, episode_size=episode_size,
                                save=save)

    attack_no_defence = Simulator(X, y, deepcopy(model), attacker=attacker, defender=None,
                                        batch_size=batch_size, episode_size=episode_size,
                                        save=save)
                                
    
    defence_no_attack = Simulator(X, y, deepcopy(model), attacker=None, defender=defender,
                                        batch_size=batch_size, episode_size=episode_size,
                                        save=save)

    attack_and_defence = Simulator(X, y, model, attacker=attacker, defender=defender,
                                        batch_size=batch_size, episode_size=episode_size,
                                        save=save)

    simulators_dict = {'regular': regular, 'only_attack': attack_no_defence,
                        'only_defense': defence_no_attack,
                        'attack_and_defense': attack_and_defence}

    simulators = _get_simulators(simulators_dict, exclude=exclude)

    all_results_X = {}
    all_results_y = {}
    all_models = {}

    for label, simulator in simulators.items():
        print(f"\n{label} SIMULATION:")

        simulator.learn_online(verbose)

        #retrieve current simulator results 
        all_results_X[label] = simulator.results['X_stream']
        all_results_y[label] = simulator.results['y_stream']
        all_models[label] = simulator.results['models']
        
    return all_results_X, all_results_y, all_models

def wrap_results(simulators):
    """Wrap results of different ran simulations.

    Args:
         simulators {dictionary}: Dictionary containing Simulator instances
                                  with keys as descriptive labels of their differences.
    """
    wrapped_results_X = {}
    wrapped_results_y = {}
    wrapped_models = {}

    for label, simulator in simulators.items():
        wrapped_results_X[label] = simulator.results['X_stream']
        wrapped_results_y[label] = simulator.results['y_stream']
        wrapped_models[label] = simulator.results['models']
        
    return wrapped_results_X, wrapped_results_y, wrapped_models


class Simulator:
    """"""
    def __init__(self, X, y, model, attacker=None,
                 defender=None, batch_size=1, episode_size=1, 
                 defender_kwargs = {}, save=False,  **kwargs):
        """"""
        
        self.X = X
        self.y = y
        self.episode_size = episode_size
        self.batch_size = batch_size
        self.model = model
        self.attacker = attacker
        self.defender = defender
        self.save = save
        self.defender_kwargs = defender_kwargs
        self.results = {'X_stream': [], 'y_stream': [], 'models': []}

    def run(self, verbose=True):
        """"""
        generator = DataLoader(self.X, self.y, batch_size = self.episode_size) #initialise data stream
        batch_queue = DataLoader(batch_size = self.batch_size) #initialise cache data loader
        
        batch_num = 0
        for (X_episode, y_episode) in generator:
            # Attacker's turn to attack
            if self.attacker:
                X_episode, y_episode = self.attacker.attack(X_episode, y_episode)

            # Defender's turn to defend
            if self.defender:
                if self.defender_kwargs["requires_model"]:
                    self.defender_kwargs["model"] = self.model
                X_episode, y_episode = self.defender.defend(X_episode, y_episode, **self.defender_kwargs)

            batch_queue.add_to_cache(X_episode, y_episode)
            
            # Online learning loop
            for (X_batch, y_batch) in batch_queue:

                self.model.step(X_batch, y_batch)

                if verbose:
                    print("Batch: {:03d} -- Loss: {:.4f}".format(
                        batch_num,
                        self.model.losses[-1],
                        ))

                batch_num += 1
                
            self.results["X_stream"].append(X_episode)
            self.results["y_stream"].append(y_episode)
            self.results["models"].append(deepcopy(self.model.state_dict()))
                
                # Postprocessor saves resultsb
                #postprocessor.cache(databatch, perturbed_databatch, model.epoch_loss)

            # Save the results to the results directory
            if self.save:
                save_pickle(self.results)
                            
