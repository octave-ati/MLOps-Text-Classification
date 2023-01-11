from imblearn.over_sampling import RandomOverSampler
import json
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import log_loss
from classifyops import data, predict, utils, evaluate
from config import config
from pathlib import Path
import optuna
import joblib

# Defining our training function, encapsulating all of the components developed prior
min_freq = 75

# Initializing a variable for best f1 value
best_f1 = 0

def train(args, df, trial=None):

  # Setup
  utils.set_seeds()
  df = df.sample(frac=1).reset_index(drop=True)
  df = data.preprocess(df, lower=True, stem=False, min_freq=min_freq)
  label_encoder = data.LabelEncoder().fit(df.tag)
  X_train, X_val, X_test, y_train, y_val, y_test = data.create_splits(
      X=df.text.to_numpy(),
      y=label_encoder.encode(df.tag))
  test_df = pd.DataFrame({"text": X_test, "tag": label_encoder.decode(y_test)})
  # TF-IDF Vectorization
  vectorizer = TfidfVectorizer(analyzer=args.analyzer, ngram_range=
                               (2, args.ngram_max_range)) #char n-grams
  X_train = vectorizer.fit_transform(X_train)
  X_val = vectorizer.transform(X_val)
  X_test = vectorizer.transform(X_test)

  # Oversample
  oversample = RandomOverSampler(sampling_strategy="all")
  X_over, y_over = oversample.fit_resample(X_train, y_train)

  # Model
  model = SGDClassifier(
      loss="log", penalty="l2", alpha = args.alpha, max_iter=100,
      learning_rate="constant", eta0=args.learning_rate,
      power_t=args.power_t, warm_start=True)

  # Training
  for epoch in range(args.num_epochs):
    model.fit(X_over,y_over)
    train_loss = log_loss(y_train, model.predict_proba(X_train))
    val_loss = log_loss(y_val, model.predict_proba(X_val))
    if not epoch%10:
            print(
                f"Epoch: {epoch:02d} | "
                f"train_loss: {train_loss:.5f}, "
                f"val_loss: {val_loss:.5f}"
            )

    # # Logging metrics
    # if not trial:
    #   mlflow.log_metrics({"train_loss": train_loss, "val_loss":val_loss},
    #                      step=epoch)

     # Pruning ==> this part is implemented in the hyperparameter optimization part
    if trial:
        trial.report(val_loss, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

  # Evaluation
  other_index = label_encoder.class_to_index["other"]
  y_prob = model.predict_proba(X_test)
  y_pred = predict.custom_predict(y_prob=y_prob, class_thresholds=args.threshold, index=other_index)
  performance = evaluate.get_metrics(
        y_true=y_test, y_pred=y_pred, classes=label_encoder.classes, df=test_df
    )

  return {
      "args": args,
      "label_encoder": label_encoder,
      "vectorizer": vectorizer,
      "model": model,
      "performance": performance
  }

# Defining our optimization objective
def objective(args, df, trial):
  global best_f1
  # Parameters to tune
  args.analayzer = trial.suggest_categorical("analyzer", ["word", "char", "char_wb"])
  args.ngram_max_range = trial.suggest_int("ngram_max_range", 3, 10)
  args.learning_rate = trial.suggest_loguniform("learning_rate", 1e-2, 1e0)
  args.power_t = trial.suggest_uniform("power_t", 0.1, 0.5)
  args.threshold["0"] = trial.suggest_uniform("threshold_0", 0.4, 0.8)
  args.threshold["1"] = trial.suggest_uniform("threshold_1", 0.4, 0.8)
  args.threshold["2"] = trial.suggest_uniform("threshold_2", 0.4, 0.8)
  # We don't optimize for the threshold for "other" since it is irrelevant
  # (our predict function returns "other" if the value is below the class threshold)

  # Training and evaluation
  artifacts = train(args=args, df=df, trial=trial)

  # Recording performance attributes
  overall_performance = artifacts["performance"]["overall"]
  print(json.dumps(overall_performance, indent=2))
  trial.set_user_attr("precision", overall_performance["precision"])
  trial.set_user_attr("recall", overall_performance["recall"])
  trial.set_user_attr("f1", overall_performance["f1"])

  #Saving artifacts
  if overall_performance["f1"] > best_f1:
    best_f1 = overall_performance["f1"]

    print(f"New best performance : {best_f1}")
    print("Saving model data")

    with open(Path(config.MODEL_DIR, "label_encoder.json"),"wb") as file:
      joblib.dump(artifacts["label_encoder"], file)
    with open(Path(config.MODEL_DIR,"vectorizer.pkl"), "wb") as file:
      joblib.dump(artifacts["vectorizer"], file)
    with open(Path(config.MODEL_DIR,"model.pkl"), "wb") as file:
      joblib.dump(artifacts["model"], file)
    with open(Path(config.MODEL_DIR,"performance.json"), "wb") as file:
      joblib.dump(artifacts["performance"], file)

  return overall_performance["f1"]