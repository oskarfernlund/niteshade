
# =============================================================================
#  IMPORTS AND DEPENDENCIES
# =============================================================================
import numpy as np
from data import DataLoader
from copy import deepcopy
from utils import save_pickle

def wrap_results(simulators: dict):
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

# =============================================================================
#  CLASSES
# =============================================================================
class Simulator():
    """Used to simulate data poisoning attacks during online learning. 
       
       Args:
        - X {np.ndarray, torch.Tensor}: stream of input data to train the model
                                        with during supervised learning.

        - y {np.ndarray, torch.Tensor}: stream of target data (labels to the inputs)
                                        to train the model with during supervised learning.

        - model {torch.nn.Module}: neural network model inheriting from torch.nn.Module to 
                                   be trained during online learning. Must present a .step()
                                   method that performs a gradient descent step on a batch 
                                   of input and target data (X_batch and y_batch). 
                                        
        - attacker {Attacker}: attacker object that presents a .attack() method with an 
                               implementation of a data poisoning attack strategy. 

        - defender {Defender}: Defender object that presents a .defend() method with an 
                               implementation of a data poisoning defense strategy.

        - batch_size {int}: batch size of model.          

        - num_episodes {int}: Number of 'episodes' that X and y span over. Here, we refer to
                              an episode as the time period over which a stream of incoming data
                              would be collected and subsequently passed on to the model to be 
                              trained.

    """
    def __init__(self, X, y, model, attacker=None, defender=None, 
                 batch_size=1, num_episodes=1, save=False) -> None:
        self.X = X
        self.y = y
        self.num_episodes = num_episodes
        self.episode_size = len(X) // num_episodes
        self.batch_size = batch_size
        self.model = model
        self.attacker = attacker
        self.defender = defender
        self.save = save

        self.results = {'X_stream': [], 'y_stream': [], 'models': []}

    def run(self, defender_kwargs = {}, attacker_kwargs = {}, verbose = True) -> None:
        """Runs a simulation of an online learning setting where, if specified, an attacker
           will 'poison' (i.e. perturb) incoming data points (from an episode) according to an 
           implemented attack strategy (i.e. .attack() method) and a defender (also, if 
           specified,) will reject points deemed perturbed by its defence strategy (i.e. 
           .defend() method). 

        Args:
            defender_kwargs {dict}: dictionary containing keyword arguments for defender .defend() method.
            attacker_kwargs {dict}: dictionary containing keyword arguments for attacker .attack() method.
            verbose {bool}: Default = True.
        """
        generator = DataLoader(self.X, self.y, batch_size = self.episode_size) #initialise data stream
        batch_queue = DataLoader(batch_size = self.batch_size) #initialise cache data loader
        
        batch_num = 0
        for (X_episode, y_episode) in generator:
            # Attacker's turn to attack
            if self.attacker:
                if attacker_kwargs["requires_model"]:
                    attacker_kwargs["model"] = self.model
                X_episode, y_episode = self.attacker.attack(X_episode, y_episode, **attacker_kwargs)

            # Defender's turn to defend
            if self.defender:
                if defender_kwargs["requires_model"]:
                    defender_kwargs["model"] = self.model

                X_episode, y_episode = self.defender.defend(X_episode, y_episode, **defender_kwargs)

            #print(" 2 ", X_episode.shape, y_episode.shape)
            batch_queue.add_to_cache(X_episode, y_episode) #add perturbed / filtered points to batch queue
            
            # Online learning loop
            for (X_batch, y_batch) in batch_queue:
                
                #take a gradient descent step
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

            # Save the results to the results directory
            if self.save:
                save_pickle(self.results)
                            
