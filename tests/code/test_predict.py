import numpy as np
import pandas as pd
import pytest

from classifyops import main, predict


@pytest.mark.parametrize(
    "y_prob, y_pred, threshold, index",
    [
        (
            [[0.8, 0.9, 0.5], [0.1, 0.4, 0.3]],
            np.array([1, 1]),
            {"0": 0, "1": 0, "2": 0},
            "other",
        ),  # Returns good predictions with null threshold
        (
            [[1, 0.9, 0.5], [0.1, 0.4, 0.7]],
            np.array([0, 2]),
            {
                "0": -5145,
                "1": -5464,
                "2": -89615,
            },
            "other",
        ),  # Returns good predictions with negative threshold
        (
            [[0.8, 0.9, 0.5], [0.1, 0.4, 0.3]],
            np.array(["other", "other"]),
            {"0": 1, "1": 1, "2": 1},
            "other",
        ),  # Returns only "other" index with threshold = 1
        (
            [[0.8, 0.9, 0.5], [0.1, 0.4, 0.3]],
            np.array(["other", "other"]),
            {"0": 8464, "1": 6545, "2": 15564},
            "other",
        ),  # Returns only "other" index with threshold > 1
        (
            [[0.8, 0.9, 0.5], [0.1, 0.4, 0.3], [0.2, 0.2, 0.25]],
            np.array([1, "other", 2]),
            {"0": 0, "1": 0.5, "2": 0.1},
            "other",
        ),  # Works with different thresholds
        (
            [[0.8, 0.8, 0.8], [0.3, 0.4, 0.4], [0.2, 0.2, 0.2]],
            np.array([0, "other", 0]),
            {"0": 0, "1": 0.5, "2": 0.1},
            "other",
        ),  # Equalities return first index
    ],
)
def test_custom_predict(y_prob, y_pred, threshold, index):
    pred = predict.custom_predict(y_prob, threshold, index)
    np.testing.assert_array_equal(y_pred, pred)

@pytest.mark.parametrize(
    "texts, target_pred",
    # Baseline predictions
    [
        (
            ['mlops is very important in project design to optimize production'],
            ['mlops']
        ),
        (
            ['This is a clear text classification project'] * 5,
            ['natural-language-processing'] * 5
        ),
        (
            ['During this project we will develop image classification models'] * 99,
            ['computer-vision'] * 99
        ),
        (
            ['no clear category for that project'] * 10,
            ['other'] * 10
        ),
    ],
)
def test_predict(texts, target_pred):
    pred = predict.predict(texts, main.load_artifacts())
    pred_df = pd.DataFrame(pred)  # Fails if columns names inconsistent
    assert len(pred) == len(texts) # One prediction per input
    # Good column names
    np.testing.assert_array_equal(pred_df.columns, ["input_text", "predicted_tags"])
    np.testing.assert_array_equal(pred_df['predicted_tags'],target_pred)
