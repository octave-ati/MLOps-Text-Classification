import json
from classifyops import utils
from config import config
from pathlib import Path
import os
import random
import numpy as np

def test_load_dict():
    # Verifying that the loaded json is a dict
    assert isinstance(utils.load_dict(Path(config.MODEL_DIR,'args.json')), dict)

def test_save_dict():
    test = {'a': 1, 'b': 2}
    # Saving dict
    utils.save_dict(test, Path(config.TEST_DIR, "test.json"))
    # Retrieving dict
    with open(Path(config.TEST_DIR, "test.json"), "rb") as fp:
        d = json.load(fp)
    # Asserting dict equality
    assert test == d
    # Deleting created dict
    os.remove(Path(config.TEST_DIR,"test.json"))

def test_set_seeds():
    # Generating base number
        utils.set_seeds()
        a = random.randint(25,50)
        b = np.random.random()

    # Regenerating seeds and comparing
        utils.set_seeds()
        assert a == random.randint(25,50)
        assert b == np.random.random()

    # Setting different seeds
        utils.set_seeds(486)
        assert a != random.randint(25,60)
        assert b != np.random.random()


