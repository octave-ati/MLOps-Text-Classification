from pathlib import Path

from google.cloud import bigquery
from google.oauth2 import service_account

from airflow.decorators import dag
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago

PROJECT_ID = "mlops-text-classif"
SERVICE_ACCOUNT_KEY_JSON = "/home/faskill/Files/0. Files/1. WIP/9. Projet R/Github Portfolio/MLOPS-Text-Classification/secrets/mlops-text-classif-817dd22281bf.json"

BASE_DIR = Path(__file__).parent.parent.parent.absolute()
DATA_DIR = Path(BASE_DIR, "data")

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

    extract_from_dwh


# Run DAG
ml = mlops()
