from fastapi import Query
from pydantic import BaseModel, validator


class Text(BaseModel):
    text: str = Query(None, min_length=1)


# Using BaseModel class to make use of built-in validation
class PredictPayload(BaseModel):
    texts: list[Text]

    @validator("texts")
    def list_must_not_be_empty(cls, value):
        if not len(value):
            raise ValueError("List of texts to classify cannot be empty.")
        return value

    # Enables running test queries within the documentation
    class Config:
        schema_extra = {
            "example": {
                "texts": [
                    {"text": "Transfer learning with transformers for text classification."},
                    {"text": "Generative adversarial networks in both PyTorch and TensorFlow."},
                ]
            }
        }
