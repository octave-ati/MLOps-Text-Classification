import json
from pathlib import Path

import joblib
import mlflow
import optuna
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import log_loss

from classifyops import data, evaluate, predict, utils
from config import config

# Defining our training function, encapsulating all of the components developed prior
min_freq = 75

# Initializing a variable for best f1 value
best_f1 = 0


def train(args: dict, df: pd.DataFrame, trial: optuna.trial.Trial = None) -> dict:
    """Trains a SGDClassifier model with the given arguments (see args).
      The training is conducted on 70% of the initial dataframe.
      Before training, the df is preprocessed and vectorized using TFIDF.
      The train set is also oversampled to ensure equity between classes.
      This function returns a dictionary with the model performance and artifacts.

    Args:
        args (dict): Arguments the model will be trained with.
          Must contain the following arguments (with example) :
          {
            "shuffle": true,
            "subset": null,
            "min_freq": 75,
            "lower": true,
            "stem": false,
            "analyzer": "char_wb",
            "ngram_max_range": 8,
            "alpha": 0.0001,
            "learning_rate": 0.0168175454255382,
            "power_t": 0.4621204423573871,
            "threshold": { #Those are class thresholds (see custom_predict in predict.py)
              "0": 0.43187984953453573,
              "1": 0.4918490095691183,
              "2": 0.5422219542982806,
              "3": 0.5
            },
            "num_epochs": 100
          }

        trial (optuna.trial.Trial, optional): Optuna trial.
          Will only be used in the case of a hyperparameter optimization.
          If no trial is selected, the artifacts will be saved to mlflow.
          For more info see optimize function below.
          Defaults to None.

    Raises:
        optuna.TrialPruned: If the trial is pruned, skips to the next trial.

    Returns:
        artifacts (dict): Dictionary containing the following elements:
              - vectorizer(model) : Character vectorizer
              - model (SGDclassifier model) : Trained prediction model
              - args (dict) : model arguments (same format as args above)
              - label_encoder (LabelEncoder()) : Custom Label Encoder instance.
              - performance (dict, Optional) : Performance of the provided model.
    """
    # Setup
    utils.set_seeds()
    df = df.sample(frac=1).reset_index(drop=True)
    df = data.preprocess(df, lower=True, stem=False, min_freq=min_freq)
    label_encoder = data.LabelEncoder().fit(df.tag)
    X_train, X_val, X_test, y_train, y_val, y_test = data.create_splits(
        X=df.text.to_numpy(), y=label_encoder.encode(df.tag)
    )
    test_df = pd.DataFrame({"text": X_test, "tag": label_encoder.decode(y_test)})
    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer(
        analyzer=args.analyzer, ngram_range=(2, args.ngram_max_range)
    )  # char n-grams
    X_train = vectorizer.fit_transform(X_train)
    X_val = vectorizer.transform(X_val)
    X_test = vectorizer.transform(X_test)

    # Oversample
    oversample = RandomOverSampler(sampling_strategy="all")
    X_over, y_over = oversample.fit_resample(X_train, y_train)

    # Model
    model = SGDClassifier(
        loss="log",
        penalty="l2",
        alpha=args.alpha,
        max_iter=100,
        learning_rate="constant",
        eta0=args.learning_rate,
        power_t=args.power_t,
        warm_start=True,
    )

    # Training
    for epoch in range(args.num_epochs):
        model.fit(X_over, y_over)
        train_loss = log_loss(y_train, model.predict_proba(X_train))
        val_loss = log_loss(y_val, model.predict_proba(X_val))
        if not epoch % 10:
            print(
                f"Epoch: {epoch:02d} | "
                f"train_loss: {train_loss:.5f}, "
                f"val_loss: {val_loss:.5f}"
            )

        # Logging metrics
        if not trial:
            mlflow.log_metrics({"train_loss": train_loss, "val_loss": val_loss}, step=epoch)

        # Pruning ==> this part is implemented in the hyperparameter optimization part
        if trial:
            trial.report(val_loss, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

    # Evaluation
    other_index = label_encoder.class_to_index["other"]
    y_prob = model.predict_proba(X_test)
    y_pred = predict.custom_predict(
        y_prob=y_prob, class_thresholds=args.threshold, index=other_index
    )
    performance = evaluate.get_metrics(
        y_true=y_test, y_pred=y_pred, classes=label_encoder.classes, df=test_df
    )

    return {
        "args": args,
        "label_encoder": label_encoder,
        "vectorizer": vectorizer,
        "model": model,
        "performance": performance,
    }


#
# Defining our optimization objective
def objective(args: dict, df: pd.DataFrame, trial: optuna.trial.Trial) -> float:
    """Objective defined to perform hyperparameter optimization using the optuna package.
      Target metric : f1 score
      Arguments to tune :
        - Character analyzer (analyzer)
        - Ngram_max_range : Defines the number of ngrams of the vectorizer
        - Learning rate
        - Power_t : The exponent for inverse scaling learning rate
        - Class thresholds for classes 0, 1 and 2 (class 3 is our "other" class)

      At the end of each step, if the model has the best f1 score :
      Saves the model to the root MODEL_DIR directory

    Args:
        args (dict): See argument format in train function above
        df (pd.DataFrame): Input dataframe
        trial (optuna.trial.Trial): Optuna trial

    Returns:
        float: F1 score of the optimization step
    """
    global best_f1
    # Parameters to tune
    args.analyzer = trial.suggest_categorical("analyzer", ["word", "char", "char_wb"])
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

    # Saving artifacts
    if overall_performance["f1"] > best_f1:
        best_f1 = overall_performance["f1"]

        print(f"New best performance : {best_f1}")
        print("Saving model data")

        artifacts["label_encoder"].save(Path(config.MODEL_DIR, "label_encoder.json"))
        with open(Path(config.MODEL_DIR, "vectorizer.pkl"), "wb") as file:
            joblib.dump(artifacts["vectorizer"], file)
        with open(Path(config.MODEL_DIR, "model.pkl"), "wb") as file:
            joblib.dump(artifacts["model"], file)
        utils.save_dict(artifacts["performance"], Path(config.MODEL_DIR, "performance.json"))

    return overall_performance["f1"]
