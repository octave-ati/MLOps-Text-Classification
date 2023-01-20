import os
import shutil
import unittest
from pathlib import Path

import mlflow
import pytest
from typer.testing import CliRunner

from classifyops import main
from config import config

runner = CliRunner()
args_fp = Path(config.TEST_DIR, "code", "test_args.json")


def delete_experiment(experiment_name: str) -> None:
    """Deleting test mlflow experiment

    Args:
        experiment_name (str): Name of the experiment to delete
    """
    client = mlflow.tracking.MlflowClient()
    experiment_id = client.get_experiment_by_name(experiment_name).experiment_id
    client.delete_experiment(experiment_id=experiment_id)


class TestELT(unittest.TestCase):
    def test_elt_data(self):
        # Testing logger
        with self.assertLogs(level="INFO") as test_logger:
            main.elt_data(dir=config.TEST_DIR)
            self.assertEqual(test_logger.output, ["INFO:root:✅ Data Saved"])

        # Testing file save
        files = ["projects.csv", "tags.csv", "labeled_projects.csv"]
        for f in files:
            assert os.path.isfile(Path(config.TEST_DIR, f))

        # Cleaning up files
        for f in files:
            os.remove(Path(config.TEST_DIR, f))
            assert os.path.isfile(Path(config.TEST_DIR, f)) is False


@pytest.mark.training
def test_optimize():
    study_name = "test_optimization"
    num_trials = 1
    test_run = True
    result = runner.invoke(
        main.app,
        [
            "optimize",
            f"--args-fp={args_fp}",
            f"--study-name={study_name}",
            f"--num-trials={num_trials}",
            f"--test-run={test_run}",
        ],
    )

    assert result.exit_code == 0

    # Clean up
    delete_experiment(experiment_name=study_name)
    # Removing mlflow trash folder to prevent study name from being locked
    # shutil.rmtree(Path(config.MODEL_DIR, '.trash'))


@pytest.mark.training
def test_train_model():
    experiment_name = "test"
    run_name = "test"
    test_run = "true"
    result = runner.invoke(
        main.app,
        [
            "train-model",
            f"--args-fp={args_fp}",
            f"--experiment-name={experiment_name}",
            f"--run-name={run_name}",
            f"--test-run={test_run}",
        ],
    )
    assert result.exit_code == 0

    # Clean up
    delete_experiment(experiment_name=experiment_name)
    # Removing mlflow trash folder to prevent study name from being locked
    shutil.rmtree(Path(config.MODEL_DIR, ".trash"))


def test_load_artifacts():
    run_id = open(Path(config.CONFIG_DIR, "run_id.txt")).read()
    artifacts = main.load_artifacts(run_id=run_id)
    assert len(artifacts) == 5


def test_predict_tag():
    text = "This project is clearly a text classification nlp project"
    result = runner.invoke(main.app, ["predict-tag", f"--text={text}"])
    assert result.exit_code == 0
