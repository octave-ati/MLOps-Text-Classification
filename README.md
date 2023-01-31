# ClassifyOps, a MLOps Text Classification Project

In this project, I apply MLOps and DataOps practices to the deployment
of a text classification model aiming at classifying ML projects in
different categories.

The different steps of this project include:

- Development of a MVP model on Jupyter Notebook
- Experiment Tracking with MLFlow
- Hyperparameter optimization with Optuna
- Project packaging and deployment on FastAPI
- Data visualization on Streamlit
- Test-driven development with Pytest - Great-Expectations (GE)
- Code cleaning and documentation ([Link](https://faskill.github.io/MLOps-Text-Classification/))
with mkdocs, flake8, isort and black.
- Data and model versioning using dvc and Github
- Implementation of CI/CD pratices with Github actions / pre-commit
- Creation of a modern data stack including Airbyte, BigQuery and dbt
- Orchestration of the DataOps and MLOps pipelines with Airflow
- Creation of a feature store with Feast

Here is a flowchart I created to explain the main processes implemented in the project :

![Project Flowchart](img/flowchart.png)

## Useful Links

* [Jupyter Notebook (MVP - Tracer Bullet)](Notebook.ipynb)
* [Packaged Project Folder](classifyops/)
* [FastAPI Folder](app/)
*

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

# Designing an Image Segmentation Model

An Image Segmentation Model designed to be used within the Computer Vision System
of a Self Driving Vehicle.

It is trained based on the [Cityscrapes Dataset](https://www.cityscapes-dataset.com/).

This model identifies different classes of objects in photos captured by a Vehicle's sensors :

- Constructions
- Nature
- Sky
- People
- Vehicle
- Object

Due to limited computing power at my disposal, the model is not able to differentiate between
different types of objects within the same class (example : it does not differentiate a truck
from a car).

The prediction API is then published on a web interface using Flask.

## Useful Links

* [Project presentation (Powerpoint)](Project_Presentation.pptx)
* [Technical Report of Model development (Word)](Technical_Report.docx)
* [Jupyter Notebook (Model training)](Notebook.ipynb)
* [Flask Deployment Folder](Deployment/)

## Screenshots

### Web Interface
![Web Interface](img/website.png)

### Model Prediction
![Raw Image](img/input.png)
![Prediction](img/prediction.png)
![True Mask](img/true.png)

### Predicts stock images
![Real Image](img/real_image.png)
![Prediction](img/real_prediction.png)

## Libraries / Packages Used

* [Tensorflow - Keras](https://www.tensorflow.org/)
* [Flask](https://flask.palletsprojects.com/en/2.2.x/)
* [Scikit-Image](https://scikit-image.org/)
* [Albumentations](https://albumentations.ai/)
* [Open CV](https://opencv.org/)
* [Segmentation Models](https://github.com/qubvel/segmentation_models)
* [SqueezeNet Keras Implementation](https://github.com/rcmalli/keras-squeezenet)
* [Bootstrap](https://getbootstrap.com/)
* Matplotlib / Seaborn
* Pandas / Numpy

## Developed By

Octave Antoni

<img src="https://avatars.githubusercontent.com/u/841669?v=4" width="20%">

[Connect with me on Linkedin](https://www.linkedin.com/in/octave-antoni/)

## License

    Copyright 2023 Octave Antoni

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
