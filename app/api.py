from http import HTTPStatus
from datetime import datetime
from functools import wraps

from fastapi import FastAPI, Request


from pathlib import Path
from config import config
from config.config import logger
from classifyops import main, predict
from app.schemas import PredictPayload


# Define application
app = FastAPI(
    title="MLOps Text Classification",
    description="Classification of machine learning projects with MLOps practices.",
    version="0.1",
)

@app.on_event("startup")
def load_artifacts(run_id: bool = False) -> None:
    global artifacts
    if run_id:
        # Retrieves artifacts from the run_id saved within the config folder
        run_id = open(Path(config.CONFIG_DIR, "run_id.txt")).read()
        artifacts = artifacts = main.load_artifacts(run_id = run_id, best = False)
    else:
        # Retrieves artifacts from optimized model
        artifacts = main.load_artifacts(run_id = "", best = True)
    logger.info("Ready for inference!")

def construct_response(f):
    """Construct a JSON response for an endpoint."""

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
    """Application health check."""
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
    """Get the performance metrics."""
    performance = artifacts["performance"]
    data = {"performance":performance.get(filter, performance)}
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": data,
    }
    return response

# Retrieve given argument
# Arguments must be in the following subset :
# shuffle
# subsetmin_freq
# lower
# staticmethodanalyzer
# ngram_max_range
# alpha
# learning_rate
# power_t
# num_epochs
# threshold {
#     "0"
#     "1"
#     "2"
#     "3"
# }
@app.get("/args/{arg}", tags=["Arguments"])
@construct_response
def _arg(request: Request, arg: str) -> dict:
    """Get a specific parameter's value used for the run."""
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
    """Get all arguments used for the run."""
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
def _predict(request: Request, payload: PredictPayload) -> dict:
    """Predict tags for a list of texts."""
    texts = [item.text for item in payload.texts]
    predictions = predict.predict(texts=texts, artifacts=artifacts)
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": {"predictions": predictions},
    }
    return response