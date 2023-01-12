from pathlib import Path
import nltk
from nltk.corpus import stopwords
import mlflow
import logging
import sys
from rich.logging import RichHandler

# Directories
BASE_DIR = Path(__file__).parent.parent.absolute()
CONFIG_DIR = Path(BASE_DIR, "config")
DATA_DIR = Path(BASE_DIR, "data")
MODEL_DIR = Path(BASE_DIR, "model")
STORES_DIR = Path(BASE_DIR, "stores")
LOGS_DIR = Path(BASE_DIR, "logs")

# Create dirs
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Assets
PROJECTS_URL = "https://raw.githubusercontent.com/Faskill/MLOps-Text-Classification/main/Data/projects.csv"
TAGS_URL = "https://raw.githubusercontent.com/Faskill/MLOps-Text-Classification/main/Data/tags.csv"

# Stopwords
nltk.download("stopwords")
STOPWORDS = stopwords.words("english")

# Tags
ACCEPTED_TAGS = ["natural-language-processing", "computer-vision", "mlops", "graph-learning"]

# Mlflow setup
mlflow.set_tracking_uri("file:///" + str(MODEL_DIR.absolute()))


# Logger config
logging_config = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "minimal": {"format": "%(message)s"},
        "detailed": {
            "format": "%(levelname)s %(asctime)s [%(name)s:%(filename)s:%(funcName)s:%(lineno)d]\n%(message)s\n"
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "stream": sys.stdout,
            "formatter": "minimal",
            "level": logging.DEBUG,
        },
        "info": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": Path(LOGS_DIR, "info.log"),
            "maxBytes": 10485760,  # 1 MB
            "backupCount": 10,
            "formatter": "detailed",
            "level": logging.INFO,
        },
        "error": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": Path(LOGS_DIR, "error.log"),
            "maxBytes": 10485760,  # 1 MB
            "backupCount": 10,
            "formatter": "detailed",
            "level": logging.ERROR,
        },
    },
    "root": {
        "handlers": ["console", "info", "error"],
        "level": logging.INFO,
        "propagate": True,
    },
}

# Logger setup
logging.config.dictConfig(logging_config)
logger = logging.getLogger()
logger.handlers[0] = RichHandler(markup=True)  # pretty formatting