from __future__ import annotations

import uuid
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union, cast

import faiss
import numpy as np
import numpy.typing as npt
import pandas as pd
from faiss import read_index, write_index
from tqdm import tqdm
from typing_extensions import Self

from vision_agent.emb import Embedder
from vision_agent.lmm import LMM

tqdm.pandas()


class DataStore:
    r"""A class to store and manage image data along with its generated metadata from an LMM."""

    def __init__(self, df: pd.DataFrame):
        r"""Initializes the DataStore with a DataFrame containing image paths and image IDs. If the image IDs are not present, they are generated using UUID4. The DataFrame must contain an 'image_paths' column.

        Args:
            df (pd.DataFrame): The DataFrame containing "image_paths" and "image_id" columns.
        """
        self.df = df
        self.lmm: Optional[LMM] = None
        self.emb: Optional[Embedder] = None
        self.index: Optional[faiss.IndexFlatIP] = None  # type: ignore
        if "image_paths" not in self.df.columns:
            raise ValueError("image_paths column must be present in DataFrame")
        if "image_id" not in self.df.columns:
            self.df["image_id"] = [str(uuid.uuid4()) for _ in range(len(df))]

    def add_embedder(self, emb: Embedder) -> Self:
        self.emb = emb
        return self

    def add_lmm(self, lmm: LMM) -> Self:
        self.lmm = lmm
        return self

    def add_column(
        self, name: str, prompt: str, func: Optional[Callable[[str], str]] = None
    ) -> Self:
        r"""Adds a new column to the DataFrame containing the generated metadata from the LMM.

        Args:
            name (str): The name of the column to be added.
            prompt (str): The prompt to be used to generate the metadata.
            func (Optional[Callable[[Any], Any]]): A Python function to be applied on the output of `lmm.generate`. Defaults to None.
        """
        if self.lmm is None:
            raise ValueError("LMM not set yet")

        self.df[name] = self.df["image_paths"].progress_apply(  # type: ignore
            lambda x: (
                func(self.lmm.generate(prompt, image=x))
                if func
                else self.lmm.generate(prompt, image=x)
            )
        )
        return self

    def build_index(self, target_col: str) -> Self:
        r"""This will generate embeddings for the `target_col` and build a searchable index over them, so next time you run search it will search over this index.

        Args:
            target_col (str): The column name containing the data to be indexed."""
        if self.emb is None:
            raise ValueError("Embedder not set yet")

        embeddings: pd.Series = self.df[target_col].progress_apply(lambda x: self.emb.embed(x))  # type: ignore
        embeddings_np = np.array(embeddings.tolist()).astype(np.float32)
        self.index = faiss.IndexFlatIP(embeddings_np.shape[1])
        self.index.add(embeddings_np)
        return self

    def get_embeddings(self) -> npt.NDArray[np.float32]:
        if self.index is None:
            raise ValueError("Index not built yet")

        ntotal = self.index.ntotal
        d: int = self.index.d
        return cast(
            npt.NDArray[np.float32],
            faiss.rev_swig_ptr(self.index.get_xb(), ntotal * d).reshape(ntotal, d),
        )

    def search(self, query: str, top_k: int = 10) -> List[Dict]:
        r"""Searches the index for the most similar images to the query and returns the top_k results.

        Args:
            query (str): The query to search for.
            top_k (int, optional): The number of results to return. Defaults to 10."""
        if self.index is None:
            raise ValueError("Index not built yet")
        if self.emb is None:
            raise ValueError("Embedder not set yet")

        query_embedding: npt.NDArray[np.float32] = self.emb.embed(query)
        _, idx = self.index.search(query_embedding.reshape(1, -1), top_k)
        return cast(List[Dict], self.df.iloc[idx[0]].to_dict(orient="records"))

    def save(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.mkdir(parents=True)
        self.df.to_csv(path / "data.csv")
        if self.index is not None:
            write_index(self.index, str(path / "data.index"))

    @classmethod
    def load(cls, path: Union[str, Path]) -> DataStore:
        path = Path(path)
        df = pd.read_csv(path / "data.csv", index_col=0)
        ds = DataStore(df)
        if Path(path / "data.index").exists():
            ds.index = read_index(str(path / "data.index"))
        return ds


def build_data_store(data: Union[str, Path, list[Union[str, Path]]]) -> DataStore:
    if isinstance(data, Path) or isinstance(data, str):
        data = Path(data)
        data_files = list(Path(data).glob("*"))
    elif isinstance(data, list):
        data_files = [Path(d) for d in data]

    df = pd.DataFrame()
    df["image_paths"] = data_files
    df["image_id"] = [uuid.uuid4() for _ in range(len(data_files))]
    return DataStore(df)
