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