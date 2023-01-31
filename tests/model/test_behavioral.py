from pathlib import Path

import pytest

from classifyops import main, predict
from config import config


@pytest.fixture(scope="module")
def artifacts():
    run_id = open(Path(config.CONFIG_DIR, "run_id.txt")).read()
    artifacts = main.load_artifacts(run_id=run_id)
    return artifacts


@pytest.mark.parametrize(
    "text, tag",
    [
        (
            "Transformers and NLP are really important in machine learning.",
            "natural-language-processing",
        ),
        (
            "Transformers and NLP are really unnecessary in machine learning.",
            "natural-language-processing",
        ),
    ],
)
def test_inv(text, tag, artifacts):
    """INVariance via adjective injection (changes should not impact prediction)."""
    predicted_tag = predict.predict(texts=[text], artifacts=artifacts)[0]["predicted_tags"]
    assert tag == predicted_tag


@pytest.mark.parametrize(
    "text, tag",
    [
        (
            "ML applied to text classification.",
            "natural-language-processing",
        ),
        (
            "ML applied to image classification.",
            "computer-vision",
        ),
        (
            "CNNs for text classification.",
            "natural-language-processing",
        ),
    ],
)
def test_dir(text, tag, artifacts):
    """DIRectional expectations (changes to key inputs)."""
    predicted_tag = predict.predict(texts=[text], artifacts=artifacts)[0]["predicted_tags"]
    assert tag == predicted_tag


@pytest.mark.parametrize(
    "text, tag",
    [
        (
            "Natural language processing is the next big wave in machine learning.",
            "natural-language-processing",
        ),
        (
            "MLOps is the next big wave in machine learning.",
            "mlops",
        ),
        (
            "Computer vision is the future of image classification",
            "computer-vision",
        ),
    ],
)
def test_mft(text, tag, artifacts):
    """Minimum Functionality Tests (simple input/output pairs)."""
    predicted_tag = predict.predict(texts=[text], artifacts=artifacts)[0]["predicted_tags"]
    assert tag == predicted_tag
