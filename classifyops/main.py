import json
import tempfile
import warnings
from argparse import Namespace
from pathlib import Path

import joblib
import mlflow
import optuna
import pandas as pd
import typer
from numpyencoder import NumpyEncoder
from optuna.integration.mlflow import MLflowCallback

from classifyops import data, predict, train, utils
from config import config
from config.config import logger

warnings.filterwarnings("ignore")

# Initializing App with typer
app = typer.Typer()


# This function will be called from the Python interpreter
@app.command()
def elt_data(dir: Path = config.DATA_DIR):
    """
    This function extracts the dataset from the github project
    It then transfers it to a local file
        Args:
        dir (Path): directory where the csv will be saved. Defaults to /data
    """
    # Extract and Load
    projects = pd.read_csv(config.PROJECTS_URL)
    tags = pd.read_csv(config.TAGS_URL)
    projects.to_csv(Path(dir, "projects.csv"), index=False)
    tags.to_csv(Path(dir, "tags.csv"), index=False)

    # Transform

    df = pd.merge(projects, tags, on="id", how="outer")
    df = df[df.tag.notnull()]  # Deleting rows without a tag
    df.to_csv(Path(dir, "labeled_projects.csv"), index=False)

    logger.info("✅ Data Saved")


# Optimizing
@app.command()
def optimize(
    args_fp: str = "config/args.json",
    study_name: str = "optimization",
    num_trials: int = 20,
    test_run: str = "false",
    direction: str = "maximize",
    metric: str = "f1",
) -> None:
    """Runs a hyperparameter optimization algorithm

    Args:
        args_fp (str): Path to the base arguments to be used at initialization
        study_name (str): Name of the MLflow study
        num_trials (int): Number of trials of the study
        test_run (bool): Set to True only during testing. Defaults to False.
        direction (str): Direction to optimize metric. Defaults to maximize.
        metric (str): Key metric to optimize. Defaults to f1 score.
    """
    # Loading labeled data
    df = pd.read_csv(Path(config.DATA_DIR, "labeled_projects.csv"))
    args = Namespace(**utils.load_dict(filepath=args_fp))

    # Defining pruner
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5)

    # Creating study
    study = optuna.create_study(study_name=study_name, direction=direction, pruner=pruner)

    # Defining callback
    mlflow_callback = MLflowCallback(tracking_uri=mlflow.get_tracking_uri(), metric_name=metric)

    # Performing study
    study.optimize(
        lambda trial: train.objective(
            args, df, trial, test_run=test_run, metric=metric, direction=direction
        ),
        n_trials=num_trials,
        callbacks=[mlflow_callback],
    )

    # Best trial
    trials_df = study.trials_dataframe()
    trials_df = trials_df.sort_values(["user_attrs_f1"], ascending=False)

    if not test_run == "true":  # pragma: no cover, Prevent argument saving during tests
        # Saving best parameters
        utils.save_dict({**args.__dict__, **study.best_trial.params}, args_fp, cls=NumpyEncoder)
        print(f"\nBest value ({metric}): {study.best_trial.value}")
        print(f"Best hyperparameters: {json.dumps(study.best_trial.params, indent=2)}")


@app.command()
def train_model(
    args_fp: str = "config/args.json",
    experiment_name: str = "baselines",
    run_name: str = "sgd",
    test_run: str = "false",
) -> None:
    """Trains the model (generally 100 epochs) and records experiment to MLflow
    The function also logs metrics and artifacts to mlflow for later retrieval

    Args:
        args_fp (str): Location of the arguments to be used within the model training
        experiment_name (str): Name of the MLflow experiment
        run_name (str): Name of the MLflow training run
        test_run (str): Set to true only during testing. Defaults to False.
    """
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
    if not test_run == "true":  # pragma: no cover, Prevent argument saving during tests
        open(Path(config.CONFIG_DIR, "run_id.txt"), "w").write(run_id)
        utils.save_dict(performance, Path(config.CONFIG_DIR, "performance.json"))


def load_artifacts(run_id: str = "", best: bool = True) -> dict:
    """Loads artifacts from a MLFlow experiment

    Args:
        run_id (str): Run ID of the MLflow experiment
        best (bool, optional): Either we retrieve the best value (stored in the root model folder)
        Or the value of the experiment with the given run_id. Defaults to True.

    Returns:
        dict:
                args(dict): Final optimized arguments
                label_encoder(model): Saved LabelEncoder
                vectorizer(model): Pickled character vectorizer
                model(model): Pickled Model
                performance(dict): Performance metrics.
    """

    # If best is True, we will recover artifacts stored in the root MODEL_DIR
    if best:
        artifacts_dir = Path(config.MODEL_DIR)
    else:  # pragma: no cover, current implementation does not save artifacts in mlflow
        # Locate specific artifacts directory
        experiment_id = mlflow.get_run(run_id=run_id).info.experiment_id
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
        "performance": performance,
    }


@app.command()
def predict_tag(text: str = "", run_id: str = None, best: bool = True) -> list[dict]:
    """Predicts a tag with the retrieved artifacts

    Args:
        text (str): Text input (that we want to predict)
        run_id (str, optional): Run_ID of the experiment we want to load assets from. Defaults to None.
        best (bool, optional): True if we want to retrieve the best performing model (F1 Score).
        False if we want to load assets from the given run_id. Defaults to True.

    Returns:
        list(dict) - single element list :
            input_text(str): Same as the text given as an argument.
            predicted_tags(str): Predicted tag from config.ACCEPTED_TAGS
    """

    if not run_id:
        run_id = open(Path(config.CONFIG_DIR, "run_id.txt")).read()
    artifacts = load_artifacts(run_id=run_id, best=best)
    prediction = predict.predict(texts=[text], artifacts=artifacts)
    print(json.dumps(prediction, indent=2))
    return prediction


if __name__ == "__main__":
    app()
