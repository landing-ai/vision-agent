import os
import platform
import shutil
from functools import lru_cache
from importlib import resources
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
import requests
from openai import AzureOpenAI, OpenAI
from scipy.spatial.distance import cosine  # type: ignore

from vision_agent.tools.tools import get_tools_df
from vision_agent.utils.tools import (
    get_vision_agent_api_key,
    _create_requests_session,
    _LND_API_URL_v2,
)


@lru_cache(maxsize=1)
def get_tool_recommender() -> "Sim":
    return load_cached_sim(get_tools_df())


@lru_cache(maxsize=512)
def get_embedding(
    emb_call: Callable[[List[str]], List[float]], text: str
) -> List[float]:
    text = text.replace("\n", " ")
    return emb_call([text])


def load_cached_sim(
    tools_df: pd.DataFrame, sim_key: str = "desc", cached_dir: str = ".sim_tools"
) -> "Sim":
    cached_dir_full_path = str(resources.files("vision_agent") / cached_dir)
    if os.path.exists(cached_dir_full_path):
        if tools_df is not None:
            if StellaSim.check_load(cached_dir_full_path, tools_df):
                # don't pass sim_key to loaded Sim object or else it will re-calculate embeddings
                return StellaSim.load(cached_dir_full_path)
    if os.path.exists(cached_dir_full_path):
        shutil.rmtree(cached_dir_full_path)

    sim = StellaSim(tools_df, sim_key=sim_key)
    sim.save(cached_dir_full_path)
    return sim


def stella_embeddings(prompts: List[str]) -> List[np.ndarray]:
    # implement stella here so we don't display embeddings which can be costly
    payload = {
        "input": prompts,
        "model": "stella1.5b",
    }
    url = f"{_LND_API_URL_v2}/embeddings"
    vision_agent_api_key = get_vision_agent_api_key()
    headers = {
        "Authorization": f"Basic {vision_agent_api_key}",
        "X-Source": "vision_agent",
    }
    if runtime_tag := os.environ.get("RUNTIME_TAG", "vision-agent"):
        headers["runtime_tag"] = runtime_tag

    session = _create_requests_session(
        url=url,
        num_retry=3,
        headers=headers,
    )
    response = session.post(url, data=payload).json()
    data = response["data"]
    return [np.array(d["embedding"]) for d in data]


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
            api_key: Optional[str]: The OpenAI API key to use for embeddings.
            model: str: The model to use for embeddingshttps://github.com/landing-ai/vision-agent/pull/280.
        """
        self.df = df
        self.client = OpenAI(api_key=api_key)
        self.emb_call = (
            lambda x: self.client.embeddings.create(input=x, model=model)
            .data[0]
            .embedding
        )
        self.model = model
        if "embs" not in df.columns and sim_key is None:
            raise ValueError("key is required if no column 'embs' is present.")

        if sim_key is not None:
            self.df = self.df.assign(
                embs=self.df[sim_key].apply(
                    lambda x: get_embedding(
                        self.emb_call,
                        x,
                    )
                )
            )

    def save(self, save_dir: Union[str, Path]) -> None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        df = self.df.copy()
        embs = np.array(df.embs.tolist())
        np.save(save_dir / "embs.npy", embs)
        df = df.drop("embs", axis=1)
        df.to_csv(save_dir / "df.csv", index=False)

    @staticmethod
    def load(
        load_dir: Union[str, Path],
        api_key: Optional[str] = None,
        model: str = "text-embedding-3-small",
    ) -> "Sim":
        load_dir = Path(load_dir)
        df = pd.read_csv(load_dir / "df.csv")
        embs = np.load(load_dir / "embs.npy")
        df["embs"] = list(embs)
        return Sim(df, api_key=api_key, model=model)

    @staticmethod
    def check_load(
        load_dir: Union[str, Path],
        df: pd.DataFrame,
    ) -> bool:
        load_dir = Path(load_dir)
        if (
            not Path(load_dir / "df.csv").exists()
            or not Path(load_dir / "embs.npy").exists()
        ):
            return False

        df_load = pd.read_csv(load_dir / "df.csv")
        if platform.system() == "Windows":
            df_load = df_load.assign(
                doc=df_load.doc.apply(lambda x: x.replace("\r", ""))
            )
        return df.equals(df_load)  # type: ignore

    @lru_cache(maxsize=256)
    def top_k(
        self,
        query: str,
        k: int = 5,
        thresh: Optional[float] = None,
    ) -> Sequence[Dict]:
        """Returns the top k most similar items to the query.

        Parameters:
            query: str: The query to compare to.
            k: int: The number of items to return.
            thresh: Optional[float]: The minimum similarity threshold.

        Returns:
            Sequence[Dict]: The top k most similar items.
        """

        embedding = get_embedding(
            self.emb_call,
            query,
        )
        self.df = self.df.assign(
            sim=self.df.embs.apply(lambda x: 1 - cosine(x, embedding))
        )
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
            self.df = self.df.assign(
                embs=self.df[sim_key].apply(
                    lambda x: get_embedding(
                        self.emb_call,
                        x,
                    )
                )
            )


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
            self.df = self.df.assign(
                embs=self.df[sim_key].apply(
                    lambda x: get_embedding(
                        self.emb_call,
                        x,
                    )
                )
            )


class StellaSim(Sim):
    def __init__(
        self,
        df: pd.DataFrame,
        sim_key: Optional[str] = None,
    ) -> None:
        self.df = df

        def emb_call(text: List[str]) -> List[float]:
            return stella_embeddings(text)[0]  # type: ignore

        self.emb_call = emb_call

        if "embs" not in df.columns and sim_key is None:
            raise ValueError("key is required if no column 'embs' is present.")

        if sim_key is not None:
            self.df = self.df.assign(
                embs=self.df[sim_key].apply(
                    lambda x: get_embedding(
                        self.emb_call,
                        x,
                    )
                )
            )

    @staticmethod
    def load(
        load_dir: Union[str, Path],
        api_key: Optional[str] = None,
        model: str = "stella1.5b",
    ) -> "StellaSim":
        load_dir = Path(load_dir)
        df = pd.read_csv(load_dir / "df.csv")
        embs = np.load(load_dir / "embs.npy")
        df["embs"] = list(embs)
        return StellaSim(df)


def merge_sim(sim1: Sim, sim2: Sim) -> Sim:
    return Sim(pd.concat([sim1.df, sim2.df], ignore_index=True))


def load_sim(sim_file: Union[str, Path]) -> Sim:
    sim_file = Path(sim_file)
    df = pd.read_csv(sim_file / "df.csv")
    embs = np.load(sim_file / "embs.npy")
    df["embs"] = list(embs)
    return Sim(df)
