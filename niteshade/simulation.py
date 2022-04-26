#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Classes and functions for simulating data poisoning attacks and defences 
against online learning.
"""


# =============================================================================
#  IMPORTS AND DEPENDENCIES
# =============================================================================

import inspect
from copy import deepcopy
from collections import defaultdict

import torch
import numpy as np
from tqdm import tqdm

from niteshade.data import DataLoader
from niteshade.attack import Attacker
from niteshade.defence import DefenderGroup, Defender
from niteshade.utils import save_pickle, copy


# =============================================================================
#  CLASSES
# =============================================================================
class _KeyMap(object):
    """Object used to convert NumPy arrays/PyTorch Tensors
       to a hashable form."""
    def __init__(self, X, y):
        #convert to numpy arrays if data are torch.tensors
        if isinstance(X, torch.Tensor) and isinstance(y, torch.Tensor):
            if X.is_cuda and y.is_cuda:
                X = X.detach().cpu().numpy()
                y = y.detach().cpu().numpy()
            else: 
                X = X.detach().numpy()
                y = y.detach().numpy()

        self.data = X
        self.target = y
        self.hash = hash((hash(self.data.tobytes()), hash(self.target.tobytes())))
    def __hash__(self):
        return self.hash
    def __str__(self):
        return f'{self.hash}'
    def __repr__(self):
        return f'{self.hash}'

class Simulator():
    """
    Class used to simulate data poisoning attacks during online learning. 
    It is important to note the distinction between batch size and num_episodes
    in simulating data poisoning attacks during online learning; the batch size 
    here refers to the size of the mini-batches the specified model is trained on, 
    while the number of episodes refers to the number of times the data
    becomes available for the model to train on (which is equivalent to the number
    of times the attacker and defender are able to intervene in the learning process).
    The default batch size and number of episodes is 1, implying that the
    model is trained one sample at a time (no mini-batches) and the data 
    comes in all at once (i.e attacker and defender only intervene once 
    in this scenario).
       
    Args:
        X (np.ndarray, torch.Tensor) : stream of input data to train the model
                                        with during supervised learning.
        y (np.ndarray, torch.Tensor) : stream of target data (labels to the inputs)
                                        to train the model with during supervised learning.
        model (torch.nn.Module) : neural network model inheriting from torch.nn.Module to 
                                    be trained during online learning. Must present a .step()
                                    method that performs a gradient descent step on a batch 
                                    of input and target data (X_batch and y_batch).                    
        attacker (Attacker) : Attacker object that presents a .attack() method with an 
                                implementation of a data poisoning attack strategy. 
        defender (Defender) : Defender object that presents a .defend() method with an 
                                implementation of a data poisoning defense strategy.
        batch_size (int) : batch size of model.          
        num_episodes (int) : Number of 'episodes' that X and y span over. Here, we refer to
                                an episode as the time period over which a stream of incoming data
                                would be collected and subsequently passed on to the model to be 
                                trained.
    """
    def __init__(self, X, y, model, attacker=None, defender=None, 
                 batch_size=1, num_episodes=1, save=False) -> None:
        #checks
        if not batch_size > 0 and batch_size <= len(X):
             raise ValueError('Batch size must be 0 < batch_size <= len(X).')
        if not num_episodes > 0 and num_episodes <= len(X):
            raise ValueError('Number of episodes must be 0 < num_episodes <= len(X).')
        if num_episodes * batch_size > len(X):
            raise ValueError("num_episodes * batch_size must be < len(X).")
        if attacker is not None:
            if not isinstance(attacker, Attacker):
                raise TypeError('Implemented attacker must inherit from abstract Attacker object.')
        if defender is not None:
            if not isinstance(defender, (Defender, DefenderGroup)):
                raise TypeError("Implemented defender/s must inherit from abstract Defender object or be a DefenderGroup.")
        if not isinstance(model, torch.nn.Module):
            raise TypeError('Niteshade only supports PyTorch models (i.e inheriting from torch.nn.Module).')
        if not (isinstance(X, (np.ndarray, torch.Tensor)) 
                and isinstance(y, (np.ndarray, torch.Tensor))):
            raise TypeError("Niteshade only supports NumPy arrays and PyTorch tensors.") 

        #miscellaneous
        self.X = X
        self.y = y
        self.num_episodes = num_episodes
        self.episode_size = len(X) // num_episodes
        self.batch_size = batch_size
        self.model = model
        self.attacker = attacker
        self.defender = defender
        self.save = save
        self.episode = 0

        #get attacker and defender args
        if attacker:
            args, defaults = self._get_func_args(self.attacker.attack)
            if defaults:
                self.true_attacker_args = args[3:-len(defaults)] # assuming first three arguments are self, X_episode, y_episode
            else: 
                self.true_attacker_args = args[3:] # assuming first three arguments are self, X_episode, y_episode

        if defender:
            args, defaults = self._get_func_args(self.defender.defend)
            if defaults:
                self.true_defender_args = args[3:-len(defaults)] # assuming first three arguments are self, X_episode, y_episode
            else: 
                self.true_defender_args = args[3:] # assuming first three arguments are self, X_episode, y_episode

        #save original, post-attacked, and post-defended points on an
        #episodic basis
        self._original_ids = {}
        self._attacked_ids = {}
        self._defended_ids = {}
        self.num_poisoned = 0
        self.num_defended = 0
        self.epoch = 0

        #logging of results
        self.results = {'original': [], 'post_attack': [], 'post_defense':[], 'models': []}
        self._cp_labels = {0:'original', 1:'post_attack', 2:'post_defense'}
    
    def _assign_ids(self, X, y):      
        """Build a dictionary using the true datapoints as keys and their indices 
           (i.e identifiers) as values.
        
        Args: 
            X (np.ndarray, torch.Tensor) : stream of input data to train the model
                                           with during supervised learning.
            y (np.ndarray, torch.Tensor) : stream of target data (labels to the inputs)
                                           to train the model with during supervised learning.
        """
        point_ids = {}
        for idx, (inpt, label) in enumerate(zip(X,y)):
            point_hash = hash(_KeyMap(inpt, label))
            point_ids[point_hash] = f'o_{idx}_{self.epoch}'
        return point_ids
    
    def _get_func_args(self, func):
        """Get the arguments of a function."""
        args, varargs, varkw, defaults = inspect.getargspec(func)
        return args, defaults
    
    def _get_valid_args(self, func_args, args):
        """Get arguments from specified attacker/defender key-word arguments 
           that are in the actual implemented .attack() / .defend() methods."""
        return [key for key in args.keys() if key in func_args]

    def _check_for_missing_args(self, input_args, is_attacker):
        """Check if any of the specified arguments for the attacker/defender
           are missing from the actual implemented .attack() / .defend() methods.
        """
        if is_attacker:
            args, defaults = self._get_func_args(self.attacker.attack)
            valid_args = self._get_valid_args(args, input_args) #get coinciding arguments 
            if not all([arg in self.true_attacker_args for arg in valid_args]):
                missing_args = [arg for arg in self.true_attacker_args if arg not in valid_args]
                raise ArgNotFoundError(f"Arguments: {missing_args} are missing in attacker_args for .attack() method.")
        else: 
            args, defaults = self._get_func_args(self.defender.defend)
            valid_args = self._get_valid_args(args, input_args) #get coinciding arguments 
            if not all([arg in self.true_defender_args for arg in valid_args]):
                missing_args = [arg for arg in self.true_defender_args if arg not in valid_args]
                raise ArgNotFoundError(f"Arguments {missing_args} are missing in defender_args for .defend() method.")
        return valid_args

    def _shape_check(self, orig_X, orig_y, X, y):
        """Checks if the shapes of the inputs or labels have been altered when 
           perturbing/rejecting datapoints during online learning.
           
           Args: 
                orig_X (np.ndarray, torch.Tensor) : original inputs. 
                orig_y (np.ndarray, torch.Tensor) : original labels. 
                X (np.ndarray, torch.Tensor) : New inputs. 
                y (np.ndarray, torch.Tensor) : New labels. 
        """
        if len(orig_y) > 1 and len(y) > 1:
            if orig_y.shape[1:] != y.shape[1:]:
                raise ShapeMismatchError(f"""Shape (dims>0) of the labels has been altered within .attack()/.defend():
                                Original shape: {orig_y.shape}
                                New shape: {y.shape}
                                            
                                ***Please note that the batch_size (dim=0) SHOULD change when 
                                perturbing/rejecting***
                                """)
        elif len(orig_X) > 1 and len(X) > 1:
            if orig_X.shape[1:] != X.shape[1:]:
                raise ShapeMismatchError(f"""Shape (dims>0) of the inputs has been altered within .attack()/.defend():
                                Original shape: {orig_X.shape}
                                New shape: {X.shape}
                                            
                                ***Please note that the batch_size (dim=0) SHOULD change when 
                                perturbing/rejecting***
                                """)  
    
    def _get_id(self, hash, checkpoint):
        """
        Get ID's of points in episode by comparing it to the points in 
        the previous checkpoint (i.e attacked points are compare to original
        to determine if a point was poisoned or not and defended points are 
        compared with attacker points to determine if a point was rejected/modified
        """
        if checkpoint == 0:
            return self._datapoint_ids[hash]
        elif checkpoint == 1:
            point_id = self._original_ids.get(hash, 'p')
            if point_id == 'p':
                point_id = f'p_{self.num_poisoned}'
                self.num_poisoned += 1
            return point_id
        elif checkpoint == 2:
            if self.attacker:
                point_id = self._attacked_ids.get(hash, 'd')
            else:
                point_id = self._original_ids.get(hash, 'd')

            if point_id == 'd':
                point_id = f'd_{self.num_defended}'
                self.num_defended += 1
            return point_id

    def _log(self, X, y, checkpoint):
        """Log the results of an episode in the results dictionary."""
        data = {}
        for inpt, label in zip(X,y):
            point_hash = hash(_KeyMap(inpt,label)) 
            point_id = self._get_id(point_hash, checkpoint)
            #record data in running dictionaries for comparison
            if checkpoint == 0:
                self._original_ids[point_hash] = point_id
            elif checkpoint == 1:
                self._attacked_ids[point_hash] = point_id
            elif checkpoint == 2:
                self._defended_ids[point_hash] = point_id

            data[point_id] = (inpt,label) #save point with id as key and (X,y) as value

        self.results[self._cp_labels[checkpoint]].append(data)

    def run(self, defender_args = {}, attacker_args = {}, attacker_requires_model=False, 
            defender_requires_model=False, shuffle=False) -> None:
        """
        Runs a simulation of an online learning setting where, if specified, an attacker
        will "poison" incoming data points in an episode according to an 
        implemented attack strategy (via its .attack() method) and a defender (also, if 
        specified) will reject/modify points deemed poisoned by its defence strategy (via
        its .defend() method). 

        **NOTE**: If the attacker/defender require a model for their attack/defense strategies, 
        the user should only set attacker_requires_model=True/defender_requires_model=True.
        The .attack()/.defend() method should then contain the argument 'model'; 
        this argument will be added as a key to attacker_args/defender_args and updated with 
        the new model after each gradient descent step as online learning progresses.

        **Metadata**: 
        
        As the simulation progresses --> each episodes original, post-attack, 
        and post-defense inputs and labels will be saved in self.results
        (a dictionary) in a list under the keys: "original", "post-attack", 
        and "post-defense", respectively. All the datapoints in an episode 
        are saved as values in a dictionary where the keys are labels indicating 
        if a point is unperturbed (in which case the label is simply the index 
        of the point in the inputted X and y), poisoned (labelled as "p_n" 
        where n is an integer indicating that it is the nth poisoned point), 
        or modified by the defender (labelled as "d_n" where n is an integer
        indicating that it is the nth defended point). If the defender rejects 
        a point in episode i, it can be inferred by inspecting the points missing
        from self.results["post_defense"][i] with respect to self.results["post_attack"][i].

        Args:
            defender_args (dict) : dictionary containing extra arguments (other than the episode inputs
                                   X and labels y) for defender .defend() method.
            attacker_args (dict) : dictionary containing extra arguments (other than the episode inputs
                                   X and labels y) for attacker .attack() method.
            attacker_requires_model (bool) : specifies if the .attack() method of the attacker requires 
                                             the updated model at each episode.
            defender_requires_model (bool) : specifies if the .defend() method of the defender requires 
                                             the updated model at each episode.
            shuffle (bool) : Boolean indicating if passed X and y should be shuffled in DataLoader.
        """
        #save original data with index/epoch combination as id's
        self._datapoint_ids = self._assign_ids(self.X, self.y)
        self.epoch += 1

        generator = DataLoader(self.X, self.y, batch_size = self.episode_size, 
                               shuffle=shuffle) #initialise data stream
        batch_queue = DataLoader(batch_size = self.batch_size) #initialise cache data loader
        
        with tqdm(generator, desc="Running simulation", unit="episode") as tepoch: 
            for episode, (X_episode, y_episode) in enumerate(tepoch):
                orig_X_episode = copy(X_episode)
                orig_y_episode = copy(y_episode)
                #save ids of true points
                self._log(X_episode, y_episode, checkpoint=0) #log results

                # Attacker's turn to attack
                if self.attacker:
                    if attacker_requires_model:
                        if "model" in self.true_attacker_args:
                            attacker_args["model"] = self.model
                        else: 
                            raise ArgNotFoundError("Argument 'model' was not found in .attack() method.")

                    if episode == 0:
                        #look at args of .attack() method to check for inconsistencies with inputted ones
                        valid_attacker_args = self._check_for_missing_args(input_args=attacker_args, is_attacker=True)
                        #use only arguments that are actually in method
                        attacker_args = {key:value for key, value in attacker_args.items() if key in valid_attacker_args}
                    
                    #pass episode datapoints to attacker
                    X_episode, y_episode = self.attacker.attack(X_episode, y_episode, **attacker_args)

                    #check if shapes have been altered in .attack() method
                    self._shape_check(orig_X_episode, orig_y_episode, X_episode, y_episode)
                    self._log(X_episode, y_episode, checkpoint=1) #log results

                # Defender's turn to defend
                if self.defender:
                    if defender_requires_model:
                        if "model" in self.true_defender_args:
                            defender_args["model"] = self.model
                        else: 
                            raise ArgNotFoundError("Argument 'model' was not found in .defend() method.")

                    if episode == 0:
                        #look at args of .attack() method to check for inconsistencies with inputted ones
                        valid_defender_args = self._check_for_missing_args(input_args=defender_args, is_attacker=False)
                        #use only arguments that are actually in method
                        defender_args = {key:value for key, value in defender_args.items() if key in valid_defender_args}

                    #pass possibly perturbed points onto defender
                    X_episode, y_episode = self.defender.defend(X_episode, y_episode, **defender_args)

                    #check if shapes have been altered in .defend() method
                    self._shape_check(orig_X_episode, orig_y_episode, X_episode, y_episode)
                    self._log(X_episode, y_episode, checkpoint=2) #log results

                batch_queue.add_to_cache(X_episode, y_episode) #add perturbed / filtered points to batch queue
                
                # Online learning loop
                running_loss = 0
                num_batches = len(batch_queue)
                for (X_batch, y_batch) in batch_queue:
                    
                    try:
                        #take a gradient descent step
                        self.model.step(X_batch, y_batch) 
                    except AttributeError:
                        raise NotImplementedError("Model must have a .step() method to perform a gradient descent step.")

                    if hasattr(self.model, 'losses'):
                        loss = self.model.losses[-1]
                        running_loss += loss.item()/len(X_batch)

                if running_loss != 0:
                    tepoch.set_postfix(loss=running_loss/num_batches)
                else:
                    tepoch.set_postfix(loss="n/a")

                #save model state dictionary
                state_dict = deepcopy(self.model.state_dict())
                self.results['models'].append(state_dict)
                self.episode += 1

                #reinitialize episode id lists for different checkpoints
                self._original_ids = {}
                self._attacked_ids = {}
                self._defended_ids = {}

        # Save the results to the results directory
        if self.save:
            save_pickle(self.results)

                            
class ArgNotFoundError(Exception):
    """Exception to be raised if a key-word argument is missing when calling 
       the .attack()/.defend() methods of the attacker/defender."""

class ShapeMismatchError(Exception):
    """Exception to be raised when there is a shape mismatch between the 
       original episode datapoints/labels and the perturbed/rejected
       datapoints/labels by the attacker/defender."""

# =============================================================================
#  FUNCTIONS
# =============================================================================

def wrap_results(simulators: dict):
    """Wrap results of different ran simulations.

    Args:
         simulators {dictionary}: Dictionary containing Simulator instances
                                  with keys as descriptive labels of their
                                  differences.
    """
    wrapped_data = defaultdict(dict)
    wrapped_models = {}

    for label, simulator in simulators.items():
        wrapped_data[label]['original'] = simulator.results['original']
        wrapped_data[label]['post_attack'] = simulator.results['post_attack']
        wrapped_data[label]['post_defense'] = simulator.results['post_defense']
        wrapped_models[label] = simulator.results['models']
    
    return wrapped_data, wrapped_models


# =============================================================================
#  MAIN ENTRY POINT
# =============================================================================

if __name__ == '__main__':
    pass