# Written by: Jaime
# Last edited: 2022/04/08
# Description: Buidling a framework for simulating data posioning attacks during online learning

# =============================================================================
#  IMPORTS AND DEPENDENCIES
# =============================================================================
import numpy as np
from data import DataLoader
import inspect
from copy import deepcopy
from utils import save_pickle
import torch
from utils import train_test_iris

def wrap_results(simulators: dict):
    """Wrap results of different ran simulations.

    Args:
         simulators {dictionary}: Dictionary containing Simulator instances
                                  with keys as descriptive labels of their
                                  differences.
    """
    wrapped_data = {}
    wrapped_models = {}

    for label, simulator in simulators.items():
        wrapped_data[label] = simulator.results['data']
        wrapped_models[label] = simulator.results['models']
        
    return wrapped_data, wrapped_models

# =============================================================================
#  CLASSES
# =============================================================================
class KeySet(object):
    """Object used to convert NumPy arrays /PyTorch Tensors
       to a hashable form."""
    def __init__(self, X, y, tr_point = None):
        #convert to numpy arrays if data are torch.tensors
        if [type(X), type(y)] == [torch.Tensor, torch.Tensor]:
            X = X.numpy()
            y = y.numpy()

        self.tr_point = tr_point
        self.data = X
        self.target = y

    def __hash__(self):
        return hash((hash(self.data.tobytes()), hash(self.target.tobytes())))
    def __str__(self):
        return f'{self.hash_val}'
    def __repr__(self):
        return f'training_point #{self.tr_point}'

class Simulator():
    """Class used to simulate data poisoning attacks during online learning. 
       
       Args:
        - X {np.ndarray, torch.Tensor}: stream of input data to train the model
                                        with during supervised learning.

        - y {np.ndarray, torch.Tensor}: stream of target data (labels to the inputs)
                                        to train the model with during supervised learning.

        - model {torch.nn.Module}: neural network model inheriting from torch.nn.Module to 
                                   be trained during online learning. Must present a .step()
                                   method that performs a gradient descent step on a batch 
                                   of input and target data (X_batch and y_batch). 
                                        
        - attacker {Attacker}: Attacker object that presents a .attack() method with an 
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

        assert batch_size > 0, 'Batch size must be greater than 0.'
        assert num_episodes > 0, 'Number of episodes must be greater than 0.'

        self.X = X
        self.y = y
        self._datapoint_ids = self._assign_ids(self.X, self.y)
        self.num_poisoned = 0
        self.num_episodes = num_episodes
        self.episode_size = len(X) // num_episodes
        self.batch_size = batch_size
        self.model = model
        self.attacker = attacker
        self.defender = defender
        self.save = save
        self.episode = 0
        self.training_point = 1

        self.results = {'data': [], 'models': []}
    
    def _assign_ids(self, X, y):      
        """Build a dictionary using the true datapoints as keys and their indices 
           (i.e identifiers) as values.
        
        Args: 
            - X {np.ndarray, torch.Tensor}: stream of input data to train the model
                                            with during supervised learning.

            - y {np.ndarray, torch.Tensor}: stream of target data (labels to the inputs)
                                            to train the model with during supervised learning.
        """
        point_ids = {}
        for idx, (inpt, label) in enumerate(zip(X,y)):
            point_hash = KeySet(inpt, label)
            point_ids[hash(point_hash)] = idx
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
        else: 
            args, defaults = self._get_func_args(self.defender.defend)

        if defaults:
            true_args = args[3:-len(defaults)] # assuming first three arguments are self, X_episode, y_episode
        else: 
            true_args = args[3:] # assuming first three arguments are self, X_episode, y_episode

        valid_args = self._get_valid_args(args, input_args) #get coinciding arguments 

        if valid_args != true_args:
            missing_args = [arg for arg in true_args if arg not in valid_args]

            if is_attacker:
                raise ArgNotFoundError(f"""Arguments: {missing_args} are missing 
                                           in attacker_args for .attack() method.""")
            else:
                raise ArgNotFoundError(f"""Arguments {missing_args} are missing 
                                           in defender_args for .defend() method.""")
        return valid_args

    def _shape_check(self, orig_X, orig_y, X, y):
        """Checks if the shapes of the inputs or labels have been altered when 
           perturbing/rejecting datapoints during online learning.
           
           Args: 
                - orig_X {np.ndarray, torch.Tensor}: original inputs. 
                - orig_y {np.ndarray, torch.Tensor}: original labels. 
                - X {np.ndarray, torch.Tensor}: New inputs. 
                - y {np.ndarray, torch.Tensor}: New labels. 
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
    
    def _log(self, X, y, state_dict):
        """Log the results of an episode in the results dictionary.
        """
        data = {}

        for idx, (inpt, label) in enumerate(zip(X,y)):
            point_hash = KeySet(inpt,label,self.training_point)
            point_id = self._datapoint_ids.get(hash(point_hash), 'p')
            if point_id == 'p':
                point_id = f'p_{self.num_poisoned}'
                self.num_poisoned += 1
            data[point_hash] = point_id
            self.training_point += 1

        self.results['data'].append(data)
        self.results['models'].append(state_dict)

    def run(self, defender_args = {}, attacker_args = {}, attacker_requires_model=False, 
            defender_requires_model=False, verbose = True) -> None:
        """Runs a simulation of an online learning setting where, if specified, an attacker
           will 'poison' (i.e. perturb) incoming data points (from an episode) according to an 
           implemented attack strategy (i.e. .attack() method) and a defender (also, if 
           specified,) will reject points deemed perturbed by its defence strategy (i.e. 
           .defend() method). 

           NOTE: 
           If the attacker/defender require a model for their attack/defense strategies, 
           the user should only set attacker_requires_model=True/defender_requires_model=True.
           The .attack()/.defend() method should then contain the argument 'model'; 
           this argument will be added as a key to attacker_kwargs/defender_kwargs and updated with 
           the new model after each gradient descent step as online learning progresses.

        Args:
            - defender_kwargs {dict}: dictionary containing extra arguments (other than the episode inputs
                                      X and labels y) for defender .defend() method.
            - attacker_kwargs {dict}: dictionary containing extra arguments (other than the episode inputs
                                      X and labels y) for attacker .attack() method.
            - attacker_requires_model {bool}: specifies if the .attack() method of the attacker requires 
                                              the updated model at each episode.
            - defender_requires_model {bool}: specifies if the .defend() method of the defender requires 
                                              the updated model at each episode.
            - verbose {bool}: Specifies if loss should be printed for each batch the model is trained on. 
                              Default = True.
        """
        self.num_poisoned = 0
        generator = DataLoader(self.X, self.y, batch_size = self.episode_size) #initialise data stream
        batch_queue = DataLoader(batch_size = self.batch_size) #initialise cache data loader
        
        batch_num = 0
        for episode, (X_episode, y_episode) in enumerate(generator):
            # Attacker's turn to attack
            if self.attacker:
                if attacker_requires_model:
                    attacker_args["model"] = self.model
                
                if episode == 0:
                    #look at kwargs of .attack() method to check for inconsistencies
                    valid_attacker_args = self._check_for_missing_args(input_args=attacker_args, is_attacker=True)
                    #use only arguments that are actually in method
                    attacker_args = {key:value for key, value in attacker_args.items() if key in valid_attacker_args}
                
                #pass episode datapoints to attacker
                orig_X_episode = X_episode.copy()
                orig_y_episode = y_episode.copy()
                X_episode, y_episode = self.attacker.attack(X_episode, y_episode, **attacker_args)

                #check if shapes have been altered in .attack() method
                self._shape_check(orig_X_episode, orig_y_episode, X_episode, y_episode)

            # Defender's turn to defend
            if self.defender:
                if defender_requires_model:
                    defender_args["model"] = self.model

                if episode == 0:
                    #look at kwargs of .attack() method to check for inconsistencies
                    valid_defender_args = self._check_for_missing_args(input_args=defender_args, is_attacker=False)
                    #use only arguments that are actually in method
                    defender_args = {key:value for key, value in defender_args.items() if key in valid_defender_args}

                #pass possibly perturbed points onto defender
                orig_X_episode = X_episode.copy()
                orig_y_episode = y_episode.copy()
                X_episode, y_episode = self.defender.defend(X_episode, y_episode, **defender_args)

                #check if shapes have been altered in .defend() method
                self._shape_check(orig_X_episode, orig_y_episode, X_episode, y_episode)

            batch_queue.add_to_cache(X_episode, y_episode) #add perturbed / filtered points to batch queue
            
            # Online learning loop
            for (X_batch, y_batch) in batch_queue:
                
                #take a gradient descent step
                
                self.model.step(X_batch, y_batch) 

                if hasattr(self.model, 'losses'):
                    loss = self.model.losses[-1]
                else: 
                    loss = None

                if verbose:
                    print("Batch: {:03d} -- Loss: {:.4f}".format(
                        batch_num,
                        loss,
                        ))

                batch_num += 1
                
            state_dict = deepcopy(self.model.state_dict())
            self._log(X_episode, y_episode, state_dict) #log results

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

if __name__ == '__main__':
    X_train, y_train, X_test, y_test = train_test_iris(num_stacks=1)

    mydict = {}
    X = X_train[0]
    y = y_train[0]

    hash_val1 = KeySet(X,y)
    mydict[hash(hash_val1)] = 1

    X = X_train[0]
    y = y_train[0]

    hash_val2 = KeySet(X,y)
    print(hash_val1)
    print(hash_val2)
    print(mydict[hash(hash_val2)])

