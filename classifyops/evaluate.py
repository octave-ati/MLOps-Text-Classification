import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
from snorkel.slicing import PandasSFApplier, slicing_function


@slicing_function()
def nlp_cnn(x):
    """Returns a slice containing projects with a NLP tag and convolution in their text

    Args:
        x (pd.DataFrame): Dataset to be sliced

    Returns:
        pd.DataFrame : Sliced dataset.
    """
    nlp_projects = "natural-language-processing" in x.tag
    convolution_projects = "CNN" in x.text or "convolution" in x.text
    return nlp_projects and convolution_projects


@slicing_function()
def short_text(x):
    """Returns a slice containing projects with less than 8 words in their description

    Args:
        x (pd.DataFrame): Dataset to be sliced

    Returns:
        pd.DataFrame : Sliced dataset.
    """
    return len(x.text.split()) < 8  # less than 8 words


def get_slice_metrics(y_true, y_pred, slices, average: str = "micro") -> dict:
    """Calculate metrics for each slices

    Args:
        y_true (pd.df or np.array): True values
        y_pred (pd.df or np.array): Predicted values
        slices (List[pd.DataFrame or np.array]): List of slices
        average (str): Average used within the precision_recall_fscore function

    Returns:
        dict: Return slice metrics including :
            - Precision
            - Recall
            - F1 score
            - Number of samples of the slice
    """
    metrics = {}
    for slice_name in slices.dtype.names:
        mask = slices[slice_name].astype(bool)
        if sum(mask):
            slice_metrics = precision_recall_fscore_support(
                y_true[mask], y_pred[mask], average=average
            )
            metrics[slice_name] = {}
            metrics[slice_name]["precision"] = slice_metrics[0]
            metrics[slice_name]["recall"] = slice_metrics[1]
            metrics[slice_name]["f1"] = slice_metrics[2]
            metrics[slice_name]["num_samples"] = len(y_true[mask])
    return metrics


def get_metrics(y_true, y_pred, classes: list, df: pd.DataFrame = None) -> dict[dict]:
    """Calculate overall and class metrics, returns slice metrics if required.

    Args:
        y_true (pd.df or np.array): True values
        y_pred (pd.df or np.array): Predicted values
        classes (list): List of classes to be considered (multiclass prediction)
        df (pd.DataFrame): Dataframe to be sliced. Defaults to None.

    Returns:
        dict[dict]: Returns for the overall dataset, for each class and each slice:
            - Precision
            - Recall
            - F1 score
            - Number of samples of the dataset / class or slice
    """
    # Performance
    metrics = {"overall": {}, "class": {}}

    # Overall metrics
    overall_metrics = precision_recall_fscore_support(y_true, y_pred, average="weighted")
    metrics["overall"]["precision"] = overall_metrics[0]
    metrics["overall"]["recall"] = overall_metrics[1]
    metrics["overall"]["f1"] = overall_metrics[2]
    metrics["overall"]["num_samples"] = np.float64(len(y_true))

    # Per-class metrics
    class_metrics = precision_recall_fscore_support(y_true, y_pred, average=None)
    for i, _class in enumerate(classes):
        metrics["class"][_class] = {
            "precision": class_metrics[0][i],
            "recall": class_metrics[1][i],
            "f1": class_metrics[2][i],
            "num_samples": np.float64(class_metrics[3][i]),
        }

    # Slice metrics
    if df is not None:
        slices = PandasSFApplier([nlp_cnn, short_text]).apply(df)
        metrics["slices"] = get_slice_metrics(y_true=y_true, y_pred=y_pred, slices=slices)

    return metrics
