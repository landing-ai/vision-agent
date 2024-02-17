import uuid
from pathlib import Path

import faiss
import numpy as np
import numpy.typing as npt
import pandas as pd
from tqdm import tqdm

from lmm_tools import LMM, Embedder

tqdm.pandas()


class Data:
    def __init__(self, df: pd.DataFrame):
        self.df = pd.DataFrame()
        self.lmm: LMM | None = None
        self.emb: Embedder | None = None
        self.index = None
        if "image_paths" not in df.columns:
            raise ValueError("image_paths column must be present in DataFrame")

    def add_embedder(self, emb: Embedder):
        self.emb = emb

    def add_lmm(self, lmm: LMM):
        self.lmm = lmm

    def add_column(self, name: str, prompt: str) -> None:
        if self.lmm is None:
            raise ValueError("LMM not set yet")

        self.df[name] = self.df["image_paths"].progress_apply(
            lambda x: self.lmm.generate(prompt, image=x)
        )

    def add_index(self, target_col: str) -> None:
        if self.emb is None:
            raise ValueError("Embedder not set yet")

        embeddings = self.df[target_col].progress_apply(lambda x: self.emb.embed(x))
        embeddings = np.array(embeddings.tolist()).astype(np.float32)
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(embeddings)

    def get_embeddings(self) -> npt.NDArray[np.float32]:
        if self.index is None:
            raise ValueError("Index not built yet")

        ntotal = self.index.ntotal
        d = self.index.d
        return faiss.rev_swig_ptr(self.index.get_xb(), ntotal * d).reshape(ntotal, d)

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        if self.index is None:
            raise ValueError("Index not built yet")
        if self.emb is None:
            raise ValueError("Embedder not set yet")

        query_embedding = self.emb.embed(query)
        _, I = self.index.search(query_embedding.reshape(1, -1), top_k)
        return self.df.iloc[I[0]].to_dict(orient="records")


def build_data(data: str | Path | list[str | Path]) -> Data:
    if isinstance(data, Path) or isinstance(data, str):
        data = Path(data)
        data_files = list(Path(data).glob("*"))
    elif isinstance(data, list):
        data_files = [Path(d) for d in data]

    df = pd.DataFrame()
    df["image_paths"] = data_files
    df["image_id"] = [uuid.uuid4() for _ in range(len(data_files))]
    return Data(df)
