from datetime import datetime
from functools import wraps
from http import HTTPStatus
from pathlib import Path

from fastapi import FastAPI, Request

from app.schemas import PredictPayload
from classifyops import main, predict
from config import config
from config.config import logger

# Define application
app = FastAPI(
    title="MLOps Text Classification",
    description="Classification of machine learning projects with MLOps practices.",
    version="0.1",
)


@app.on_event("startup")
def load_artifacts(run_id: bool = False) -> None:
    """Load artifacts either from a run_id or from our best artifacts

    Args:
        run_id (bool, optional): If False, retrieves best artifacts.
            If True, retrieves artifacts from the run_id in config/run_id.txt.
            Defaults to False.
    """
    global artifacts
    if run_id:
        # Retrieves artifacts from the run_id saved within the config folder
        run_id = open(Path(config.CONFIG_DIR, "run_id.txt")).read()
        artifacts = artifacts = main.load_artifacts(run_id=run_id, best=False)
    else:
        # Retrieves artifacts from optimized model
        artifacts = main.load_artifacts(run_id="", best=True)
    logger.info("Ready for inference!")


def construct_response(f):
    """Wrapper function to Generate a complete response from API Endpoint
        Adds the CRUD method, status-code, timestamp and request URL

    Args:
        f (function): Input function to be wrapped

    Returns:
        _type_: Wrapper function
    """

    @wraps(f)
    def wrap(request: Request, *args, **kwargs) -> dict:
        results = f(request, *args, **kwargs)
        response = {
            "message": results["message"],
            "method": request.method,
            "status-code": results["status-code"],
            "timestamp": datetime.now().isoformat(),
            "url": request.url._url,
        }
        if "data" in results:
            response["data"] = results["data"]
        return response

    return wrap


@app.get("/", tags=["General"])
@construct_response
def _index(request: Request) -> dict:
    """Application health check

    Args:
        request (Request): Inbound request

    Returns:
        dict: Returns a standard everything is OK phrase and a OK HTTPStatus
    """
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": {},
    }
    return response


# Retrieve performance data
@app.get("/performance", tags=["Performance"])
@construct_response
def _performance(request: Request, filter: str = None) -> dict:
    """Returns the performance metrics of the loaded model

    Args:
        request (Request): Inbound request
        filter (str, optional): Filter to retrieve metrics for.
            Can be one of : overall, class, slices
            Defaults to None.

    Returns:
        dict: Performance either for all 3 types (overall, class and slices)
            Or for the filtered type
    """
    performance = artifacts["performance"]
    data = {"performance": performance.get(filter, performance)}
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": data,
    }
    return response


@app.get("/args/{arg}", tags=["Arguments"])
@construct_response
def _arg(request: Request, arg: str) -> dict:
    """Retrieve requested argument from the loaded model

    Args:
        request (Request): Inbound request
        arg (str): Argument to retrieve.
            Must be in the following subset :
            shuffle, min_freq, lower, analyzer
            ngram_max_range, alpha, learning_rate
            power_t, num_epochs, threshold

    Returns:
        dict: Value of the requested argument
    """
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": {
            arg: vars(artifacts["args"]).get(arg, ""),
        },
    }
    return response


# Retrieve all arguments
@app.get("/args", tags=["Arguments"])
@construct_response
def _args(request: Request) -> dict:
    """Returns all arguments used in the currently loaded model

    Args:
        request (Request): Inbound request

    Returns:
        dict: List of arguments, refer to the _arg function for the complete list
    """
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": {
            "args": vars(artifacts["args"]),
        },
    }
    return response


@app.post("/predict", tags=["Prediction"])
@construct_response
def _predict(request: Request, payload: PredictPayload) -> list[dict]:
    """Predict the tags for a list of utterances

    Args:
        request (Request): Inbound request
        payload (PredictPayload): Data for the request must be in the following format:
            "texts": [
            {"text": "Example test 1."},
            {"text": "Example test 2."}
        ]

    Returns:
        list[dict]: Prediction for each inbound text
            Dict composed of input_text and predicted_tag
    """
    texts = [item.text for item in payload.texts]
    predictions = predict.predict(texts=texts, artifacts=artifacts)
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": {"predictions": predictions},
    }
    return response
