""" 
Main abstraction layer for I/O operations. The aim is to provide a library that
more than doing many complicated things, does one and (tries) does it well.
This acts as a single source of truth as well since if someone wants to add
files, folders or change their name just has to come here and change them.

Future
`MAIN_FOLDER` & `folders` might become functions, so that the relative path
generation can be more dynamic and dependent on their usage. In this way we
might even devise `folders` as both a search and generative function, so that
if we change the order of folders and/or files we can use the function to get
them instead of hardcoding their path in the code. Of course this means unique
names for files and some kind of enforcement.
"""

import json
from os.path import join
import joblib
import numpy as np
import os
import shutil
import pandas as pd

# The idea is to declaratively encode the fs tree that we have to create and 
# manage as a data structure. In this case a dict will do.

# This represents the `root` of the directory structure. Note that this means
# that the MAIN_FOLDER starts from the path we're into when importing this
MAIN_FOLDER = join(os.getcwd(), 'experiment_files')

# Every one of these dicts represent a folder and the containing files.
# The model is tree-like -> MAIN_FOLDER/{model,results,shap}/{files}
model = {
    'data_info': 'data_info.json',
    'dfTrain': 'dfTrain.csv',
    'dfTrain_without_valid': 'dfTrain_without_valid.csv',
    'dfValid': 'dfValid.csv',
    'dfTest': 'dfTest.csv',
    'model': 'model.pkl',
    'params': 'model_configuration.json',
    'scores': 'models_scores.json',
    'validation_model_no_history': 'model_no_history.json',
    'validation_model_only_aggr_act_hist': 'model_aggr_hist.json',
    'validation_model_1_event_info': 'model_1_event_info.json',
    'validation_model_2_events_info': 'model_2_events_info.json',
    'validation_model_3_events_info': 'model_3_events_info.json',
    'validation_model_4_events_info': 'model_4_events_info.json',
    'validation_model_5_events_info': 'model_5_events_info.json',
    'validation_model_6_events_info': 'model_6_events_info.json',
    'validation_model_7_events_info': 'model_7_events_info.json',
    'validation_model_8_events_info': 'model_8_events_info.json'
}

results = {
    'running': 'results_running.csv',
    'results_completed': 'results_completed.csv',
    'completed': 'completed.csv',
    'explanations_completed': 'explanations_completed.json',
    'explanations_histogram': 'explanation_histogram.json',
    'explanations_running': 'explanations_running.json',
    'mean': 'completedMean.json',
    'scores': 'scores.json',
}

shap = {
    'background': 'background.npy',
    'shap_test': 'shapley_values_test_{}.npy',
    'timestep': 'timestep_histogram_{}.json',
    'heatmap': 'shap_heatmap_{}.json',
    'running': 'shapley_values_running.npy',
}

def path_maker(parent, d):
    """Helper to generate a full usable relative path starting from a dict 
    representation of a given folder.

    Args:
        parent (str): the name of the parent folder
        d (dict): a dict representation of a folder

    Returns:
        dict: the resulting dict with nice relative paths

    Examples:
    >>> path_maker('folder', {'a': 'file.txt'})
    {'a': '/Users/user/experiment_files/folder/file.txt'}
    """    
    return {k: join(MAIN_FOLDER, parent, v) for k, v in d.items()}

folders = {
    'model': path_maker('model', model),
    'results': path_maker('results', results),
    'shap': path_maker('shap', shap),
    'plots': {},
}

##### READERS ######
""" A collection of specific readers for various types of files. """

def read_json(path, **kwargs):
    with open(path) as j:
        return json.load(j)

def read_pickle(path, **kwargs):
    with open(path, 'rb') as f:
        return joblib.load(f)

def read_numpy(path, **kwargs):
    return np.load(path)

def read_csv(path, **kwargs):
    return pd.read_csv(path, low_memory=False, **kwargs)

def read_txt(path, **kwargs):
    with open(path) as f:
        ls = f.readlines()
        d = {}
        for l in ls:
            k, v = l.split(':')
            d[k.strip()] = v.strip()
        return d

# we can see this as an object with several methods, it's just simpler and less
# verbose than a class
reader = {
    'json': read_json,
    'pkl': read_pickle,
    'pickle': read_pickle,
    'npy': read_numpy,
    'csv': read_csv,
}

def read(filename, readfn=None, **kwargs):
    """Generic read function, the normal usage is to pass the filename that we
    want to read and expect the result. This works based on files extensions,
    if the file we want to read doesn't have an extension or has a different one
    than the ones normally used we can pass a `readfn` that will be used to read
    that file.

    Args:
        filename (str): the path to the file that we want to read
        readfn (function, optional): a reading function if we want to
         control in which way to open and read the given file. Defaults to None.

    Returns:
        any: depends on the file read
    """    
    if readfn:
        return readfn(filename, **kwargs)
    else:
        ext = filename.split('.')[-1]
        return reader[ext](filename, **kwargs)

##### WRITERS ######
""" A collection of specific writers for various types of files. """

def write_json(data, path):
    with open(path, 'w') as j:
        json.dump(data, j)

def write_pickle(data, path):
    with open(path, 'wb') as f:
        joblib.dump(data, f)

def write_numpy(data, path):
    np.save(path, data)

def write_csv(data, path):
    data.to_csv(path, index=False)

writer = {
    'json': write_json,
    'pkl': write_pickle,
    'pickle': write_pickle,
    'npy': write_numpy,
    'csv': write_csv,
}

def write(data, filename, writefn=None):
    """Generic write function, the normal usage is to pass the data and the
     filename that we want to write. This works based on files extensions,
    if the file we want to write doesn't have an extension or has a different
     one than the ones normally used we can pass a `readfn` that will be used 
     to write to that file.

    Args:
        data (any): any writable data structure
        filename (str): path to the file we want to write to
        writefn (function, optional): a writing function. Defaults to None.

    Returns:
        str: the passed filename
    """    
    if writefn:
        writefn(data, filename)
        return filename
    else:
        ext = filename.split('.')[-1]
        writer[ext](data, filename)
        return filename

##### UTILITIES #####
""" A collection of useful facilities. """

def safe_mkdir(path, safe=True):
    """Create a directory SAFELY, meaning that we try to remove the directory
    to be created and all of its contents before generating it.

    Args:
        path (str): the path to the directory to create
        safe (bool, optional): whether we want it to be safe or not. Defaults to True.

    Returns:
        str: the given path
    """    
    if safe:
        try:
            shutil.rmtree(path)
        except FileNotFoundError as e:
            print('Directory {} does not exist -> Created'.format(path))
    os.makedirs(path, exist_ok=not safe)
    return path

def create_folders(folders, safe=True):
    """Safely create many folders at once taking them from the keys of a dict.
    Basically take the keys from the passed `folders` dict and use them as dir
    names.

    Args:
        folders (dict): a dict like folders above
        safe (bool, optional): whether to do it safely. Defaults to True.

    Returns:
        list: the names of the created folders
    """    
    return [
        safe_mkdir(join(MAIN_FOLDER, f), safe) 
        for f in folders.keys()
    ]