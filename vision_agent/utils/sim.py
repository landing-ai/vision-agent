from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
from openai import Client
from scipy.spatial.distance import cosine  # type: ignore


def get_embedding(
    client: Client, text: str, model: str = "text-embedding-3-small"
) -> List[float]:
    text = text.replace("\n", " ")
    return client.embeddings.create(input=[text], model=model).data[0].embedding


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
        if not api_key:
            self.client = Client()
        else:
            self.client = Client(api_key=api_key)

        self.model = model
        if "embs" not in df.columns and sim_key is None:
            raise ValueError("key is required if no column 'embs' is present.")

        if sim_key is not None:
            self.df["embs"] = self.df[sim_key].apply(
                lambda x: get_embedding(self.client, x, model=self.model)
            )

    def save(self, sim_file: Union[str, Path]) -> None:
        sim_file = Path(sim_file)
        sim_file.mkdir(parents=True, exist_ok=True)

        df = self.df.copy()
        embs = np.array(df.embs.tolist())
        np.save(sim_file / "embs.npy", embs)
        df = df.drop("embs", axis=1)
        df.to_csv(sim_file / "df.csv", index=False)

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

        embedding = get_embedding(self.client, query, model=self.model)
        self.df["sim"] = self.df.embs.apply(lambda x: 1 - cosine(x, embedding))
        res = self.df.sort_values("sim", ascending=False).head(k)
        if thresh is not None:
            res = res[res.sim > thresh]
        return res[[c for c in res.columns if c != "embs"]].to_dict(orient="records")


def merge_sim(sim1: Sim, sim2: Sim) -> Sim:
    return Sim(pd.concat([sim1.df, sim2.df], ignore_index=True))


def load_sim(sim_file: Union[str, Path]) -> Sim:
    sim_file = Path(sim_file)
    df = pd.read_csv(sim_file / "df.csv")
    embs = np.load(sim_file / "embs.npy")
    df["embs"] = list(embs)
    return Sim(df)
