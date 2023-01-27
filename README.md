# MLOPS Text Classification

(Documentation Link)[https://faskill.github.io/MLOps-Text-Classification/classifyops/main/]

### Virtual environment creation

```bash
python3 -m venv venv
source venv/bin/activate
python3 -m pip --upgrade pip
python3 -m pip install pip setuptools wheel
python3 -m pip install -e .
```

### Extracting data from the source

```bash
python3 -m pip install numpy pandas pretty-errors
python
>>from classifyops import main
```


### Launching App

uvicorn app.api:app --host 0.0.0.0 --port 8000 --reload --reload-dir classifyops --reload-dir app  # dev
gunicorn -c app/gunicorn.py -k uvicorn.workers.UvicornWorker app.api:app  # prod

# Building docker container
docker build -t classifyops:latest -f Dockerfile .

# Running container
docker run -p 8000:8000 --name classifyops classifyops:latest

# See running containers
docker ps

# Stopping container
docker stop CONTAINER_ID # To be edited with container_id
docker stop $(docker ps -a -q) # Stop all containers

# Airflow configuration
export AIRFLOW_HOME=${PWD}/airflow
AIRFLOW_VERSION=2.5.1
PYTHON_VERSION="$(python --version | cut -d " " -f 2 | cut -d "." -f 1-2)"
CONSTRAINT_URL="https://raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}/constraints-${PYTHON_VERSION}.txt"

# Installing Airflow
python3 -m pip install "apache-airflow==${AIRFLOW_VERSION}" --constraint "${CONSTRAINT_URL}"

# Initializing DB (SQLite by default)
airflow db init

# Updating airflow container with airflow.cfg
airflow db reset -y

# Initializing airflow user
airflow users create \
    --username admin \
    --firstname FIRSTNAME \
    --lastname LASTNAME \
    --role Admin \
    --email EMAIL

# Launching airflow web server
export AIRFLOW_HOME=${PWD}/airflow
airflow webserver --port 8080  # http://localhost:8080

# Launching airflow scheduler
source venv/bin/activate
export AIRFLOW_HOME=${PWD}/airflow
export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES
airflow scheduler
