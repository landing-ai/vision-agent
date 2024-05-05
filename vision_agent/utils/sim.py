from pathlib import Path
from typing import List, Optional, Sequence, Union, Dict

import pandas as pd
from openai import Client
from scipy.spatial.distance import cosine

client = Client()


def get_embedding(text: str, model: str = "text-embedding-3-small") -> List[float]:
    text = text.replace("\n", " ")
    return client.embeddings.create(input=[text], model=model).data[0].embedding


class Sim:
    def __init__(
        self,
        df: pd.DataFrame,
        key: Optional[str] = None,
        model: str = "text-embedding-3-small",
    ) -> None:
        self.df = df
        self.model = model
        if "embs" not in df.columns and key is None:
            raise ValueError("key is required if no column 'embs' is present.")

        if key is not None:
            self.df["embs"] = self.df[key].apply(
                lambda x: get_embedding(x, model=self.model)
            )

    def save(self, sim_file: Union[str, Path]) -> None:
        self.df.to_csv(sim_file, index=False)

    def top_k(self, query: str, k: int = 5) -> Sequence[Dict]:
        embedding = get_embedding(query, model=self.model)
        self.df["sim"] = self.df.embs.apply(lambda x: 1 - cosine(x, embedding))
        res = self.df.sort_values("sim", ascending=False).head(k)
        return res[[c for c in res.columsn if c != "embs"]].to_dict(orient="records")


def load_sim(sim_file: Union[str, Path]) -> Sim:
    df = pd.read_csv(sim_file)
    return Sim(df)
