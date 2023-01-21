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
