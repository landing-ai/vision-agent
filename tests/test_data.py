import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lmm_tools.data import DataStore, build_data_store


@pytest.fixture(autouse=True)
def clean_up():
    yield
    for p in Path(".").glob("test_save*"):
        if p.is_dir():
            shutil.rmtree(p)
        else:
            p.unlink()


@pytest.fixture
def small_ds():
    df = pd.DataFrame({"image_paths": ["path1", "path2"]})
    ds = DataStore(df)
    return ds


@pytest.fixture
def small_ds_with_index(small_ds):
    small_ds.add_embedder(TestEmb())
    small_ds.add_lmm(TestLMM())
    small_ds.add_column("test", "test prompt")
    small_ds.build_index("test")
    return small_ds


class TestLMM:
    def generate(self, _, **kwargs):
        return "test"


class TestEmb:
    def embed(self, _):
        return np.random.randn(128).astype(np.float32)


def test_initialize_data_store(small_ds):
    assert small_ds is not None
    assert "image_id" in small_ds.df.columns
    assert "image_paths" in small_ds.df.columns


def test_initialize_data_store_with_no_data():
    df = pd.DataFrame({"x": ["path1"]})
    with pytest.raises(ValueError):
        DataStore(df)


def test_build_data_store():
    ds = build_data_store(["path1", "path2"])
    assert isinstance(ds, DataStore)
    assert "image_id" in ds.df.columns


def test_add_index_no_emb(small_ds):
    with pytest.raises(ValueError):
        small_ds.build_index("test")


def test_add_column_no_lmm(small_ds):
    with pytest.raises(ValueError):
        small_ds.add_column("test", "test prompt")


def test_search(small_ds_with_index):
    results = small_ds_with_index.search("test", top_k=1)
    assert len(results) == 1


def test_save(small_ds_with_index):
    small_ds_with_index.save("test_save")

    assert Path("test_save").exists()
    assert Path("test_save/data.csv").exists()
    assert Path("test_save/data.index").exists()


def test_load(small_ds_with_index):
    small_ds_with_index.save("test_save")

    second_ds = DataStore.load("test_save")
    assert second_ds.df.equals(small_ds_with_index.df)
    assert second_ds.index is not None
