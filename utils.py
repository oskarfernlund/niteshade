import os
from datetime import datetime
import pickle

def get_time_stamp_as_string():
    """"""
    # Return current timestamp in a saving friendly format.
    date_time = datetime.now()
    date_time_str = date_time.strftime("%d_%b_%Y_(%H_%M_%S.%f)")
    return date_time_str


def save_pickle(results, dir_name='output', file_name='default'):
    """"""
    # Check if dirrectory exists, if not make one
    if not os.path.isdir(dir_name):
        os.mkdir(dir_name)

    # Generate plot name, if needed
    if file_name == 'default':
        file_name = f'{get_time_stamp_as_string()}'

    pickle.dump(results, open(f"{dir_name}/{file_name}", "wb"))


def load_model(filename):
    """Load a binary file containing a neural network.

    Returns:
        model {nn.Module}: Trained model inheriting nn.Module with learnable parameters .
    """
    # If you alter this, make sure it works in tandem with save_regressor
    with open(filename, 'rb') as target:
        model = pickle.load(target)
    
    return model

def save_model(model, filename):
    """Save classifier as a binary .pickle file.

    Args:
            filename {str}: name of file to save model in (excluding extension).
    """
    # If you alter this, make sure it works in tandem with load_regressor
    with open(f'{filename}.pickle', 'wb') as target:
        pickle.dump(model, target)
