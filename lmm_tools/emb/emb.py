from abc import ABC, abstractmethod
from typing import cast

import numpy as np
import numpy.typing as npt


class Embedder(ABC):
    @abstractmethod
    def embed(self, text: str) -> npt.NDArray[np.float32]:
        pass


class SentenceTransformerEmb(Embedder):
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)

    def embed(self, text: str) -> npt.NDArray[np.float32]:
        return cast(
            npt.NDArray[np.float32],
            self.model.encode([text]).flatten().astype(np.float32),
        )


class OpenAIEmb(Embedder):
    def __init__(self, model_name: str = "text-embedding-3-small"):
        from openai import OpenAI

        self.client = OpenAI()
        self.model_name = model_name

    def embed(self, text: str) -> npt.NDArray[np.float32]:
        response = self.client.embeddings.create(input=text, model=self.model_name)
        return np.array(response.data[0].embedding).astype(np.float32)


def get_embedder(name: str) -> Embedder:
    if name == "sentence-transformer":
        return SentenceTransformerEmb()
    elif name == "openai":
        return OpenAIEmb()
    else:
        raise ValueError(
            f"Unknown embedder name: {name}, currently support sentence-transformer, openai."
        )
