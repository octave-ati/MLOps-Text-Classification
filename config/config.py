from pathlib import Path
import nltk
from nltk.corpus import stopwords
import mlflow

# Directories
BASE_DIR = Path(__file__).parent.parent.absolute()
CONFIG_DIR = Path(BASE_DIR, "config")
DATA_DIR = Path(BASE_DIR, "data")
MODEL_DIR = Path(BASE_DIR, "model")
STORES_DIR = Path(BASE_DIR, "stores")

# Create dirs
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

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