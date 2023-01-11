import pandas as pd
from pathlib import Path
from argparse import Namespace
import warnings
import logging
import json
from config import config
from classifyops import utils, data, train, predict
import mlflow
from numpyencoder import NumpyEncoder
import optuna
from optuna.integration.mlflow import MLflowCallback
import joblib
import tempfile

warnings.filterwarnings("ignore")

logger = logging.getLogger()

# This function will be called from the Python interpreter
def elt_data():
    # Extract and Load
    projects = pd.read_csv(config.PROJECTS_URL)
    tags = pd.read_csv(config.TAGS_URL)
    projects.to_csv(Path(config.DATA_DIR, "projects.csv"), index=False)
    tags.to_csv(Path(config.DATA_DIR, "tags.csv"), index=False)

    # Transform

    df = pd.merge(projects, tags, on="id", how="outer")
    df = df[df.tag.notnull()] # Deleting rows without a tag
    df.to_csv(Path(config.DATA_DIR, "labeled_projects.csv"), index=False)

    logger.info("✅ Data Saved")

# Optimizing hyperparameters
def optimize(args_fp, study_name, num_trials):
    # Loading labeled data
    df = pd.read_csv(Path(config.DATA_DIR, "labeled_projects.csv"))
    args = Namespace(**utils.load_dict(filepath=args_fp))

    # Defining pruner
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5)

    # Creating study
    study = optuna.create_study(study_name=study_name, direction="maximize", pruner=pruner)

    # Defining callback
    mlflow_callback = MLflowCallback(
        tracking_uri=mlflow.get_tracking_uri(), metric_name="f1")


    # Performing study
    study.optimize(
        lambda trial: train.objective(args, df, trial),
        n_trials=num_trials,
        callbacks=[mlflow_callback])

    # Best trial
    trials_df = study.trials_dataframe()
    trials_df = trials_df.sort_values(["user_attrs_f1"], ascending=False)
    utils.save_dict({**args.__dict__, **study.best_trial.params}, args_fp, cls=NumpyEncoder)
    print(f"\nBest value (f1): {study.best_trial.value}")
    print(f"Best hyperparameters: {json.dumps(study.best_trial.params, indent=2)}")

def train_model(args_fp, experiment_name, run_name):
    # Loading labeled data
    df = pd.read_csv(Path(config.DATA_DIR, "labeled_projects.csv"))

    # Train
    args = Namespace(**utils.load_dict(filepath=args_fp))
    mlflow.set_experiment(experiment_name=experiment_name)
    with mlflow.start_run(run_name=run_name):
        run_id = mlflow.active_run().info.run_id
        print(f"Run ID: {run_id}")
        artifacts = train.train(df=df, args=args)
        performance = artifacts["performance"]
        print(json.dumps(performance, indent=2))

        # Log metrics and parameters
        performance = artifacts["performance"]
        mlflow.log_metrics({"precision": performance["overall"]["precision"]})
        mlflow.log_metrics({"recall": performance["overall"]["recall"]})
        mlflow.log_metrics({"f1": performance["overall"]["f1"]})
        mlflow.log_params(vars(artifacts["args"]))

        # Log artifacts
        with tempfile.TemporaryDirectory() as dp:
            artifacts["label_encoder"].save(Path(dp, "label_encoder.json"))
            joblib.dump(artifacts["vectorizer"], Path(dp, "vectorizer.pkl"))
            joblib.dump(artifacts["model"], Path(dp, "model.pkl"))
            utils.save_dict(performance, Path(dp, "performance.json"))
            mlflow.log_artifacts(dp)

    # Save to config
    open(Path(config.CONFIG_DIR, "run_id.txt"), "w").write(run_id)
    utils.save_dict(performance, Path(config.CONFIG_DIR, "performance.json"))

def load_artifacts(run_id, best=True):
    """Load artifacts for a given run_id."""
    # Locate specifics artifacts directory
    experiment_id = mlflow.get_run(run_id=run_id).info.experiment_id

    # If best is True, we will recover artifacts stored in the root MODEL_DIR
    if best:
        artifacts_dir = Path(config.MODEL_DIR)
    else:
        artifacts_dir = Path(config.MODEL_DIR, experiment_id, run_id, "artifacts")

    # Load objects from run
    args = Namespace(**utils.load_dict(filepath=Path(artifacts_dir, "args.json")))
    vectorizer = joblib.load(Path(artifacts_dir, "vectorizer.pkl"))
    label_encoder = data.LabelEncoder.load(fp=Path(artifacts_dir, "label_encoder.json"))
    model = joblib.load(Path(artifacts_dir, "model.pkl"))
    performance = utils.load_dict(filepath=Path(artifacts_dir, "performance.json"))

    return {
        "args": args,
        "label_encoder": label_encoder,
        "vectorizer": vectorizer,
        "model": model,
        "performance": performance
    }

def predict_tag(text, run_id=None):
    """Predict tag for text."""
    if not run_id:
        run_id = open(Path(config.CONFIG_DIR, "run_id.txt")).read()
    artifacts = load_artifacts(run_id=run_id)
    prediction = predict.predict(texts=[text], artifacts=artifacts)
    print(json.dumps(prediction, indent=2))
    return prediction
