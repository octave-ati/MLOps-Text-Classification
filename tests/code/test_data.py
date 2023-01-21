import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from classifyops import data


@pytest.fixture(scope="module")
def df():
    data = [
        {"title": "a0", "description": "b0", "tag": "c0"},
        {"title": "a1", "description": "b1", "tag": "c1"},
        {"title": "a2", "description": "b2", "tag": "c1"},
        {"title": "a3", "description": "b3", "tag": "c2"},
        {"title": "a4", "description": "b4", "tag": "c2"},
        {"title": "a5", "description": "b5", "tag": "c2"},
        {"title": "a5", "description": "b5", "tag": "c3"},
        {"title": "a5", "description": "b5", "tag": "c3"},
        {"title": "a5", "description": "b5", "tag": "c3"},
        {"title": "a5", "description": "b5", "tag": "c3"},
    ]
    df = pd.DataFrame(data * 10)
    return df


@pytest.mark.parametrize(
    "labels, unique_labels",
    [
        ([], ["other"]),  # no set of approved labels
        (["c4"], ["other"]),  # no overlap b/w approved/actual labels
        (["c0"], ["c0", "other"]),  # partial overlap
        (["c0", "c1", "c2", "c3"], ["c0", "c1", "c2", "c3"]),  # complete overlap
    ],
)
def test_replace_oos_labels(df, labels, unique_labels):
    replaced_df = data.replace_oos_labels(
        df=df.copy(), labels=labels, label_col="tag", oos_label="other"
    )
    assert set(replaced_df.tag.unique()) == set(unique_labels)


@pytest.mark.parametrize(
    "min_freq, unique_labels",
    [
        (0, ["c0", "c1", "c2", "c3"]),  # no minimum frequency
        (-50, ["c0", "c1", "c2", "c3"]),  # negative minimum frequency
        (10, ["c0", "c1", "c2", "c3"]),  # Exact number of first label
        (11, ["c1", "c2", "c3", "other"]),  # One label removed
        (21, ["c2", "c3", "other"]),  # 2 labels removed
        (40, ["c3", "other"]),  # Only one label kept
        (41, ["other"]),  # no initial label kept
        (5465465, ["other"]),  # High number
    ],
)
def test_replace_minority_labels(df, min_freq, unique_labels):
    replaced_df = data.replace_minority_labels(
        df=df.copy(), label_col="tag", min_freq=min_freq, new_label="other"
    )
    assert set(replaced_df.tag.unique()) == set(unique_labels)


@pytest.mark.parametrize(
    "text, lower, stem, stopwords, cleaned_text",
    [
        ("Hello worlds", False, False, ["test"], "Hello worlds"),
        ("Hello worlds", True, False, ["test"], "hello worlds"),
        ("Hello Darkness my Old Friend", True, False, ["my"], "hello darkness old friend"),
        (
            "It is important to be very pythonly while you are pythoning with python. All pythoners have pythoned poorly at least once.",
            False,
            True,
            ["test"],
            "It is import to be veri pythonli while you are python with python All python have python poorli at least onc",
        ),
        ("My dad is a nice lil ol man", False, False, [], "My dad nice lil ol man"),
    ],
)
# An empty list of stopwords defaults to nltk's default stopwords by design
def test_clean_text(text, lower, stem, stopwords, cleaned_text):
    assert (
        data.clean_text(
            text=text,
            lower=lower,
            stem=stem,
            stopwords=stopwords,
        )
        == cleaned_text
    )


def test_preprocess():
    _df = pd.DataFrame({"title": "thats ", "description": "life", "tag": "c0"}, index=[0])
    assert (
        data.preprocess(
            df=_df,
            lower=False,  # Tested in test_clean_text
            stem=False,  # Tested in test_clean_text
            min_freq=0,  # Tested in test_replace_minority_labels
            labels=["c0"],  # Tested in test_replace_oos_labels
            stopwords=["dozakpdzadalsq"],  # Empty stopwords defaults to nltk EN stopwords
        )["text"][0]
        == "thats life"
    )


class TestLabelEncoder:
    @classmethod
    def setup_class(cls):
        """Called before class initialization."""
        pass

    @classmethod
    def teardown_class(cls):
        """Called after class initialization."""
        pass

    def setup_method(self):
        """Called before each method."""
        self.label_encoder = data.LabelEncoder()

    def teardown_method(self):
        """Called after each method."""
        del self.label_encoder

    def test_empty_init(self):
        label_encoder = data.LabelEncoder()
        assert label_encoder.index_to_class == {}
        assert len(label_encoder.classes) == 0

    def test_dict_init(self):
        class_to_index = {"red": 0, "blue": 1}
        label_encoder = data.LabelEncoder(class_to_index=class_to_index)
        assert label_encoder.index_to_class == {0: "red", 1: "blue"}
        assert len(label_encoder.classes) == 2

    def test_len(self):
        assert len(self.label_encoder) == 0

    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as dp:
            fp = Path(dp, "label_encoder.json")
            self.label_encoder.save(fp=fp)
            label_encoder = data.LabelEncoder.load(fp=fp)
            assert len(label_encoder.classes) == 0

    def test_str(self):
        assert str(data.LabelEncoder()) == "<LabelEncoder(num_classes=0)>"

    def test_fit(self):
        label_encoder = data.LabelEncoder()
        label_encoder.fit(["red", "red", "blue"])
        assert "red" in label_encoder.class_to_index
        assert "blue" in label_encoder.class_to_index
        assert len(label_encoder.classes) == 2

    def test_encode_decode(self):
        class_to_index = {"red": 0, "blue": 1}
        y_encoded = [0, 0, 1]
        y_decoded = ["red", "red", "blue"]
        label_encoder = data.LabelEncoder(class_to_index=class_to_index)
        print(label_encoder.decode(y_encoded))
        print(label_encoder.encode(y_decoded))
        assert np.array_equal(label_encoder.encode(y_decoded), np.array(y_encoded))
        assert label_encoder.decode(y_encoded) == y_decoded


@pytest.mark.parametrize(
    "X, y, train_size, train_count, val_count",
    [
        (np.zeros(20), np.zeros(20), 0.5, 10, 5),
        (np.zeros(100), np.zeros(100), 0.8, 80, 10),
        (np.zeros(5), np.zeros(5), 1, 1, 2),
    ],
)
def test_create_split(X, y, train_size, train_count, val_count):
    X_train, X_val, X_test, y_train, y_val, y_test = data.create_splits(X, y, train_size=train_size)
    assert len(X_train) == len(y_train) == train_count
    assert len(X_val) == len(y_val) == len(X_test) == len(y_test) == val_count
