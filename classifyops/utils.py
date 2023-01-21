import json
import random

import numpy as np


# Loading dictionary from JSON filepath
def load_dict(filepath) -> dict:
    """Open file path and returns a dict object

    Args:
        filepath (pathlib.Path or str): Path to the object in json format

    Returns:
        dict: Output dictionary
    """
    with open(filepath) as fp:
        d = json.load(fp)
    return d


# Saving dictionary to given location
def save_dict(d: dict, filepath, cls=None, sortkeys: bool = False):
    """Saving dictionary to json format

    Args:
        d (dict): Input dict object
        filepath (pathlib.Path or str): Path where we want to save the object
            The directory must be already created
        cls (JSONEnoder(), optional): Custom json encoder if required. Defaults to None.
        sortkeys (bool, optional): If True, the output dictionary will be sorted by key.
            Defaults to False.
    """
    with open(filepath, "w") as fp:
        json.dump(d, indent=2, fp=fp, cls=cls, sort_keys=sortkeys)


def set_seeds(seed: int = 42):
    """Sets numpy and random seeds.

    Args:
        seed (int, optional): Random state. Defaults to 42.
    """
    np.random.seed(seed)
    random.seed(seed)
