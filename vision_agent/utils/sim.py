import os
from functools import lru_cache
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
import requests
from openai import AzureOpenAI, OpenAI
from scipy.spatial.distance import cosine  # type: ignore


@lru_cache(maxsize=512)
def get_embedding(
    emb_call: Callable[[List[str]], List[float]], text: str
) -> List[float]:
    text = text.replace("\n", " ")
    return emb_call([text])


class Sim:
    def __init__(
        self,
        df: pd.DataFrame,
        sim_key: Optional[str] = None,
        api_key: Optional[str] = None,
        model: str = "text-embedding-3-small",
    ) -> None:
        """Creates a similarity object that can be used to find similar items in a
        dataframe.

        Parameters:
            df: pd.DataFrame: The dataframe to use for similarity.
            sim_key: Optional[str]: The column name that you want to use to construct
                the embeddings.
            model: str: The model to use for embeddings.
        """
        self.df = df
        client = OpenAI(api_key=api_key)
        self.emb_call = (
            lambda text: client.embeddings.create(input=text, model=model)
            .data[0]
            .embedding
        )
        self.model = model
        if "embs" not in df.columns and sim_key is None:
            raise ValueError("key is required if no column 'embs' is present.")

        if sim_key is not None:
            self.df["embs"] = self.df[sim_key].apply(
                lambda x: get_embedding(self.emb_call, x)
            )

    def save(self, sim_file: Union[str, Path]) -> None:
        sim_file = Path(sim_file)
        sim_file.mkdir(parents=True, exist_ok=True)

        df = self.df.copy()
        embs = np.array(df.embs.tolist())
        np.save(sim_file / "embs.npy", embs)
        df = df.drop("embs", axis=1)
        df.to_csv(sim_file / "df.csv", index=False)

    @lru_cache(maxsize=256)
    def top_k(
        self, query: str, k: int = 5, thresh: Optional[float] = None
    ) -> Sequence[Dict]:
        """Returns the top k most similar items to the query.

        Parameters:
            query: str: The query to compare to.
            k: int: The number of items to return.
            thresh: Optional[float]: The minimum similarity threshold.

        Returns:
            Sequence[Dict]: The top k most similar items.
        """

        embedding = get_embedding(self.emb_call, query)
        self.df["sim"] = self.df.embs.apply(lambda x: 1 - cosine(x, embedding))
        res = self.df.sort_values("sim", ascending=False).head(k)
        if thresh is not None:
            res = res[res.sim > thresh]
        return res[[c for c in res.columns if c != "embs"]].to_dict(orient="records")


class AzureSim(Sim):
    def __init__(
        self,
        df: pd.DataFrame,
        sim_key: Optional[str] = None,
        api_key: Optional[str] = None,
        api_version: str = "2024-02-01",
        azure_endpoint: Optional[str] = None,
        model: Optional[str] = None,
    ) -> None:
        if not api_key:
            api_key = os.getenv("AZURE_OPENAI_API_KEY")
        if not azure_endpoint:
            azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        if not model:
            model = os.getenv("AZURE_OPENAI_EMBEDDING_MODEL_DEPLOYMENT_NAME")

        if not api_key:
            raise ValueError("Azure OpenAI API key is required.")
        if not azure_endpoint:
            raise ValueError("Azure OpenAI endpoint is required.")
        if not model:
            raise ValueError(
                "Azure OpenAI embedding model deployment name is required."
            )

        self.df = df
        client = AzureOpenAI(
            api_key=api_key, api_version=api_version, azure_endpoint=azure_endpoint
        )
        self.emb_call = (
            lambda text: client.embeddings.create(input=text, model=model)
            .data[0]
            .embedding
        )

        self.model = model
        if "embs" not in df.columns and sim_key is None:
            raise ValueError("key is required if no column 'embs' is present.")

        if sim_key is not None:
            self.df["embs"] = self.df[sim_key].apply(lambda x: get_embedding(client, x))


class OllamaSim(Sim):
    def __init__(
        self,
        df: pd.DataFrame,
        sim_key: Optional[str] = None,
        model_name: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> None:
        self.df = df
        if base_url is None:
            base_url = "http://localhost:11434/api/embeddings"
        if model_name is None:
            model_name = "mxbai-embed-large"

        def emb_call(text: List[str]) -> List[float]:
            resp = requests.post(
                base_url, json={"prompt": text[0], "model": model_name}
            )
            return resp.json()["embedding"]  # type: ignore

        self.emb_call = emb_call

        if "embs" not in df.columns and sim_key is None:
            raise ValueError("key is required if no column 'embs' is present.")

        if sim_key is not None:
            self.df["embs"] = self.df[sim_key].apply(
                lambda x: get_embedding(emb_call, x)
            )


def merge_sim(sim1: Sim, sim2: Sim) -> Sim:
    return Sim(pd.concat([sim1.df, sim2.df], ignore_index=True))


def load_sim(sim_file: Union[str, Path]) -> Sim:
    sim_file = Path(sim_file)
    df = pd.read_csv(sim_file / "df.csv")
    embs = np.load(sim_file / "embs.npy")
    df["embs"] = list(embs)
    return Sim(df)
