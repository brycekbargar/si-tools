import gc
import typing
from pathlib import Path
from uuid import uuid4

import polars as pl

from .hive_dataset import HiveDataset


class Dataset:
    def __init__(self, dataset: str) -> None:
        self._dataset = dataset

    def partition_by(self, *args: str) -> HiveDataset:
        return HiveDataset(self._dataset, *args)

    def read(
        self,
        base: Path,
        low_memory: bool = False,  # noqa: FBT001, FBT002
        how: typing.Literal["vertical", "diagonal"] = "vertical",
    ) -> pl.LazyFrame:
        frames = [
            pl.scan_parquet(f, low_memory=low_memory)
            for f in (base / self._dataset).glob(str(Path("**") / "*.parquet"))
        ]

        if len(frames) == 0:
            return pl.LazyFrame()

        # https://github.com/pola-rs/polars/issues/12508
        return pl.concat(frames, how=how)

    def write(
        self,
        base: Path,
        frame: pl.LazyFrame,
    ) -> None:
        (base / self._dataset).mkdir(mode=0o755, parents=True, exist_ok=True)
        partition = frame.clone()
        partition.sink_parquet(
            base / self._dataset / f"{uuid4()}-0.parquet",
            maintain_order=False,
        )
        del partition
        gc.collect()
