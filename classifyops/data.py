import json
import re
from collections import Counter

import nltk
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config import config

stemmer = nltk.stem.PorterStemmer()
nltk.download("stopwords")


def replace_oos_labels(
    df: pd.DataFrame, labels: list, label_col: str, oos_label: str = "other"
) -> pd.DataFrame:
    """Replacing out of scope (OOS) labels with predefined label

    Args:
        df (pd.DataFrame): Input dataframe
        labels (list): List of labels within defined scope
        label_col (str): Columns where labels are located.
        oos_label (str, optional): How to name out of scope labels. Defaults to "other".

    Returns:
        pd.DataFrame: Output dataframe with modified OOS labels
    """
    oos_tags = [item for item in df[label_col].unique() if item not in labels]
    df[label_col] = df[label_col].apply(lambda x: oos_label if x in oos_tags else x)
    return df


def replace_minority_labels(
    df: pd.DataFrame, label_col: str, min_freq: int, new_label: str = "other"
) -> pd.DataFrame:
    """Replacing minority labels with predefined label

    Args:
        df (pd.DataFrame): Input dataframe
        label_col (str): Column where labels are located
        min_freq (int): Minimal label count. All labels with a count below min_freq will be replaced with new_label
        new_label (str, optional): Name of the replacement label. Defaults to "other".

    Returns:
        pd.DataFrame: Output dataframe with modified minority labels.
    """
    labels = Counter(df[label_col].values)
    labels_above_freq = Counter(label for label in labels.elements() if (labels[label] >= min_freq))
    df[label_col] = df[label_col].apply(lambda label: label if label in labels_above_freq else None)
    df[label_col] = df[label_col].fillna(new_label)
    return df


def clean_text(
    text: str, lower: bool = True, stem: bool = False, stopwords: list[str] = config.STOPWORDS
) -> str:
    """Performing following text preprocessing process:
        - Transformation to lower case (if lower=True)
        - Stopwords removal (from stopwords argument)
        - Removing non alphanumeric characters
        - Removing URLs
        - Stemming with nltk base PorterStemmer(if stem=True)

    Args:
        text (str): Input text.
        lower (bool, optional): Does the function lower the text?. Defaults to True.
        stem (bool, optional): Does the function use stemming?. Defaults to False.
        stopwords (list, optional): List of stopwords to be used. Defaults to config.STOPWORDS.

    Returns:
        str: Transformed string.
    """
    # Lower
    if lower:
        text = text.lower()

    # If there are no stopwords, use nltk's default English stopwords
    if not len(stopwords):
        stopwords = nltk.corpus.stopwords.words("english")

    # Remove stopwords
    pattern = re.compile(r"\b(" + r"|".join(stopwords) + r")\b\s*")
    text = pattern.sub("", text)

    # Spacing and filters
    text = re.sub(
        r"([!\"'#$%&()*\+,-./:;<=>?@\\\[\]^_`{|}~])", r" \1 ", text
    )  # add spacing between objects to be filtered
    text = re.sub("[^A-Za-z0-9]+", " ", text)  # remove non alphanumeric chars
    text = re.sub(" +", " ", text)  # remove multiple spaces
    text = text.strip()  # strip white space at the ends

    # Remove links
    text = re.sub(r"http\S+", "", text)

    # Stemming
    if stem:
        text = " ".join([stemmer.stem(word, to_lowercase=lower) for word in text.split(" ")])

    return text


def preprocess(
    df: pd.DataFrame,
    lower: bool,
    stem: bool,
    min_freq: int,
    labels: list = config.ACCEPTED_TAGS,
    stopwords: list = config.STOPWORDS,
) -> pd.DataFrame:
    """Preprocessing the data
    See clean_text, replace_oos_labels and replace_minority labels functions
    for more detail.

    Args:
        df (pd.DataFrame): Input dataframe
        lower (bool): If True, turns text into lowercase
        stem (bool): If True, stems text
        min_freq (int): Minimum frequency of tags kept.
        labels (list): List of accepted labels. Defaults to config.ACCEPTED_TAGS
        stopwords (list[str]): List of stopwords. Defaults to []

    Returns:
        pd.DataFrame: Preprocessed dataframe
    """
    df["text"] = df.title + " " + df.description  # feature engineering
    df.text = df.text.apply(clean_text, lower=lower, stem=stem)  # clean text
    df = replace_oos_labels(
        df=df, labels=config.ACCEPTED_TAGS, label_col="tag", oos_label="other"
    )  # replace OOS labels
    df = replace_minority_labels(
        df=df, label_col="tag", min_freq=min_freq, new_label="other"
    )  # replace labels below min freq

    return df


# Defining custom LabelEncoder
class LabelEncoder(object):
    """Defining a custom Label Encoder (based on sklearn's implementation)"""

    def __init__(self, class_to_index={}):
        self.class_to_index = class_to_index or {}  # mutable defaults ;)
        self.index_to_class = {v: k for k, v in self.class_to_index.items()}
        self.classes = list(self.class_to_index.keys())

    def __len__(self):
        return len(self.class_to_index)

    def __str__(self):
        return f"<LabelEncoder(num_classes={len(self)})>"

    def fit(self, y):
        classes = np.unique(y)
        for i, class_ in enumerate(classes):
            self.class_to_index[class_] = i
        self.index_to_class = {v: k for k, v in self.class_to_index.items()}
        self.classes = list(self.class_to_index.keys())
        return self

    def encode(self, y):
        encoded = np.zeros((len(y)), dtype=int)
        for i, item in enumerate(y):
            encoded[i] = self.class_to_index[item]
        return encoded

    def decode(self, y):
        classes = []
        for i, item in enumerate(y):
            classes.append(self.index_to_class[item])
        return classes

    def save(self, fp):
        with open(fp, "w") as fp:
            contents = {"class_to_index": self.class_to_index}
            json.dump(contents, fp, indent=4, sort_keys=False)

    @classmethod
    def load(cls, fp):
        with open(fp, "rb") as fp:
            kwargs = json.load(fp=fp)
        return cls(**kwargs)


# Creating a function to quickly define test/train/val splits
def create_splits(X, y, train_size: float = 0.7):
    """This function automatically creates train, test and validation splits.
    Train split ratio is calculated from the train_size parameter.
    Test and validation split are then created from the remaining dataset
    With a ratio of 0.5

    Args:
        X, y: Allowed inputs are lists, numpy arrays or pandas dataframes.
        train_size (float, optional): Train split ratio. Defaults to 0.7.

    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test.
    """
    X_train, X_, y_train, y_ = train_test_split(X, y, train_size=train_size, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_, y_, train_size=0.5, stratify=y_)
    return X_train, X_val, X_test, y_train, y_val, y_test
