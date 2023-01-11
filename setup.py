from pathlib import Path
from setuptools import find_namespace_packages, setup

# Loading packages from requirements.txt
BASE_DIR = Path(__file__).parent
with open(Path(BASE_DIR, "requirements.txt"), "r") as file:
    required_packages = [ln.strip() for ln in file.readlines()]

setup(
    name="classifyops",
    version=0.1,
    description="Classify ML projects.",
    author="Octave Antoni",
    author_email="octave.antoni@gmail.com",
    url="https://github.com/Faskill/MLOps-Text-Classification",
    python_requires=">=3.9",
    install_requires=[required_packages],
)