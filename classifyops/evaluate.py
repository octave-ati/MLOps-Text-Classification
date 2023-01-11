import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from snorkel.slicing import PandasSFApplier
from snorkel.slicing import slicing_function

@slicing_function()
def nlp_cnn(x):
    """NLP Projects that use convolution."""
    nlp_projects = "natural-language-processing" in x.tag
    convolution_projects = "CNN" in x.text or "convolution" in x.text
    return (nlp_projects and convolution_projects)


@slicing_function()
def short_text(x):
    """Projects with short titles and descriptions."""
    return len(x.text.split()) < 8  # less than 8 words


def get_slice_metrics(y_true, y_pred, slices):
    """Generate metrics for slices of data."""
    metrics = {}
    for slice_name in slices.dtype.names:
        mask = slices[slice_name].astype(bool)
        if sum(mask):
            slice_metrics = precision_recall_fscore_support(
                y_true[mask], y_pred[mask], average="micro"
            )
            metrics[slice_name] = {}
            metrics[slice_name]["precision"] = slice_metrics[0]
            metrics[slice_name]["recall"] = slice_metrics[1]
            metrics[slice_name]["f1"] = slice_metrics[2]
            metrics[slice_name]["num_samples"] = len(y_true[mask])
    return metrics

def get_metrics(y_true, y_pred, classes, df=None):
    """Performance metrics using ground truths and predictions."""
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
        metrics["slices"] = get_slice_metrics(
            y_true=y_true, y_pred=y_pred, slices=slices)

    return metrics
