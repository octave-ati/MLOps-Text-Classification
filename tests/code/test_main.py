from pathlib import Path
from classifyops import main
from config import config
import os
import unittest
from typer.testing import CliRunner
import mlflow
import pytest

runner = CliRunner()
args_fp = Path(config.TEST_DIR, "code","test_args.json")


def delete_experiment(experiment_name:str) -> None:
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
        with self.assertLogs(level='INFO') as test_logger:
            main.elt_data(dir=config.TEST_DIR)
            self.assertEqual(test_logger.output, ["INFO:root:✅ Data Saved"])

        # Testing file save
        files = ['projects.csv', 'tags.csv', 'labeled_projects.csv']
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
    result = runner.invoke(
        main.app,
        [
            "optimize",
            f"--args-fp={args_fp}",
            f"--study-name={study_name}",
            f"--num-trials={num_trials}",
        ],
    )
    assert result.exit_code == 0
    assert "Best value (f1):" in result.stdout
    assert "Best hyperparameters:" in result.stdout

    # Clean up
    delete_experiment(experiment_name=study_name)




