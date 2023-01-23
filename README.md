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
