import numpy as np

# Custom predict function
def custom_predict(y_prob, threshold, index):
  """Custom predict function that defaults
  to an index if conditions are not met."""
  y_pred = [np.argmax(p) if max(p) > threshold[str(np.argmax(p))] else index for p in
  y_prob]
  return np.array(y_pred)


def predict(texts, artifacts):
    """Predict tags for given texts."""
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