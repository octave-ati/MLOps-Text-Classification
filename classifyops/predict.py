import numpy as np

# Custom predict function
def custom_predict(y_prob, threshold: dict, index: int) -> np.array:
    """Returns y_prob if the probability is higher than the class threshold.

    Args:
        y_prob (np.array): Prediction probabilities
        threshold (dict): Class thresholds. If the probability is above the threshold,
        The class will be predicted, otherwise the index argument will be return.
        index (int): Index of the "other" class.

    Returns:
        np.array: Custom predictions
    """
    y_pred = [np.argmax(p) if max(p) > threshold[str(np.argmax(p))] else index for p in
    y_prob]
    return np.array(y_pred)


def predict(texts: List[str], artifacts) -> list[dict]:
    """Predict tags from a list of strings with provided artifacts

    Args:
        texts (List[str]): List of utterances
        artifacts (dict): Dictionary containing the following elements:
            - vectorizer(model) : Character vectorizer
            - model (SGDclassifier model) : Trained prediction model
            - args (dict) : model arguments
            - label_encoder (LabelEncoder()) : Custom Label Encoder instance.
            - performance (dict, Optional) : Performance of the provided model.

    Returns:
        list[dict]: List of dictionaries containing:
            - input_text(str) : Input utterance
            - predicted_tags(str) : Predicted label(s)
    """
    x = artifacts["vectorizer"].transform(texts)
    y_pred = custom_predict(
        y_prob=artifacts["model"].predict_proba(x),
        threshold=artifacts["args"].threshold,
        index=artifacts["label_encoder"].class_to_index["other"])
    tags = artifacts["label_encoder"].decode(y_pred)
    predictions = [
        {
            "input_text": texts[i],
            "predicted_tags": tags[i],
        }
        for i in range(len(tags))
    ]
    return predictions