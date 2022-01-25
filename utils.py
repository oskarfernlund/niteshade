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

    
