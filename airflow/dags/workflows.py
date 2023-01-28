import sys
from pathlib import Path

from google.cloud import bigquery
from google.oauth2 import service_account
from great_expectations_provider.operators.great_expectations import (
    GreatExpectationsOperator,
)

from airflow.decorators import dag
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago

BASE_DIR = Path(__file__).parent.parent.parent.absolute()
sys.path.append(str(BASE_DIR))

from classifyops import main

PROJECT_ID = "mlops-text-classif"
SERVICE_ACCOUNT_KEY_JSON = "/home/faskill/Files/0. Files/1. WIP/9. Projet R/Github Portfolio/MLOPS-Text-Classification/secrets/mlops-text-classif-817dd22281bf.json"


DATA_DIR = Path(BASE_DIR, "data")
GE_DIR = Path(BASE_DIR, "tests", "great_expectations")
MODEL_DIR = Path(BASE_DIR, "stores", "model")

# Default DAG args
default_args = {
    "owner": "airflow",
    "catch_up": False,
}


def _extract_from_dwh():
    """Extract labeled data from
    our BigQuery data warehouse and
    save it locally."""
    # Establish connection to DWH
    credentials = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_KEY_JSON)
    client = bigquery.Client(credentials=credentials, project=PROJECT_ID)

    # Query data
    query_job = client.query(
        """
        SELECT *
        FROM mlops_classif.labeled_projects"""
    )
    results = query_job.result()
    results.to_dataframe().to_csv(Path(DATA_DIR, "labeled_projects.csv"), index=False)


@dag(
    dag_id="mlops",
    description="MLOps tasks",
    default_args=default_args,
    schedule_interval=None,
    start_date=days_ago(2),
    tags=["mlops"],
)
def mlops():
    """MLOps workflows"""
    extract_from_dwh = PythonOperator(
        task_id="extract_data",
        python_callable=_extract_from_dwh,
    )

    validate = GreatExpectationsOperator(
        task_id="validate",
        checkpoint_name="labeled_projects",
        data_context_root_dir=GE_DIR,
        fail_task_on_validation_failure=True,
    )

    optimize = PythonOperator(
        task_id="optimize",
        python_callable=main.optimize,
        op_kwargs={
            "args_fp": Path(MODEL_DIR, "args.json"),
            "study_name": "optimization",
            "num_trials": 1,
        },
    )
    train = PythonOperator(
        task_id="train",
        python_callable=main.train_model,
        op_kwargs={
            "args_fp": Path(MODEL_DIR, "args.json"),
            "experiment_name": "baselines",
            "run_name": "sgd",
        },
    )

    extract_from_dwh >> validate >> optimize >> train


# Run DAG
ml = mlops()
