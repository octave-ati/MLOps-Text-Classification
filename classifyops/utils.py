
import json
import numpy as np
import random

# Loading dictionary from JSON filepath
def load_dict(filepath):
    with open(filepath, "r") as fp:
        d = json.load(fp)
    return d

# Saving dictionary to given location
def save_dict(d, filepath, cls=None, sortkeys=False):
    with open(filepath, "w") as fp:
        json.dump(d, indent=2, fp=fp, cls=cls, sort_keys=sortkeys)

def set_seeds(seed=42):
    # Set seeds
    np.random.seed(seed)
    random.seed(seed)