import numpy as np
import pytest
import pandas as pd
from classifyops import evaluate
from snorkel.slicing import slice_dataframe

@pytest.fixture(scope="module")
def df():
    # Additional "a"s have been added to circumvent the short slice
    data = [
        {"text": "CNN stuff on text a a a a a", "tag": "natural-language-processing"},
        {"text": "basic text classification a a a a a a", "tag": "natural-language-processing"},
        {"text": "Short", "tag": "computer-vision"},
        {"text": "Image segmentation techniques for self driving vehicles a a", "tag": "computer-vision"},
        {"text": "That project cannot be fitted into a category a a a ", "tag": "other"},
        {"text": "Data engineering practices to improve ML projects a a a a a", "tag": "mlops"}
    ]
    df = pd.DataFrame(data * 10)
    return df

def test_slicing(df):
    short_slice_df = slice_dataframe(df, evaluate.short_text)
    cnn_slice_df = slice_dataframe(df, evaluate.nlp_cnn)
    pd.testing.assert_frame_equal(
        cnn_slice_df, df[df.text == "CNN stuff on text a a a a a"]
    )
    pd.testing.assert_frame_equal(
        short_slice_df, df[df.text == "Short"]
    )

def test_get_slice_metrics():
    y_true = np.array([0,0,1,0,1,1,1,1])
    y_pred = np.array([0,0,1,1,1,1,1,1])
    slices = np.array([(1,0), (1,0), (1,0), (1,1),
        (0,1),(0,1),(0,1),(0,1)], dtype=[('a', '<i8'), ('b', '<i8')])
    metrics = evaluate.get_slice_metrics(
            y_true, y_pred, slices)
    assert metrics['a']["precision"] == 1/2
    assert metrics['a']["recall"] == 1
    assert metrics['a']["f1"] == 2*0.5/1.5
    assert metrics['a']["num_samples"] == 4
    assert metrics['b']["precision"] == 4/5
    assert metrics['b']["recall"] == 1
    assert metrics['b']["f1"] == 2*0.8/1.8
    assert metrics['b']["num_samples"] == 5

def test_get_metrics():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 0, 1])
    classes = ["a", "b"]
    performance = evaluate.get_metrics(y_true=y_true, y_pred=y_pred, classes=classes, df=None)
    assert performance["overall"]["precision"] == 2/4
    assert performance["overall"]["recall"] == 2/4
    assert performance["class"]["a"]["precision"] == 1/2
    assert performance["class"]["a"]["recall"] == 1/2
    assert performance["class"]["b"]["precision"] == 1/2
    assert performance["class"]["b"]["recall"] == 1/2