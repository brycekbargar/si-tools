import gc
import typing
from pathlib import Path
from uuid import uuid4

import polars as pl


class KeyMismatchError(Exception):
    pass


class HiveDataset:
    def __init__(self, dataset: str, *args: str) -> None:
        self._dataset = dataset
        if len(args) == 0:
            msg = "No partition keys provided"
            raise KeyMismatchError(msg)
        self._keys = args

    def read(
        self,
        base: Path,
        low_memory: bool = False,  # noqa: FBT001, FBT002
        how: typing.Literal["vertical", "diagonal"] = "vertical",
        **kwargs: typing.Any,
    ) -> pl.LazyFrame:
        path = Path(*[f"{k}={kwargs.get(k, '*')}" for k in self._keys])

        frames = []
        for f in (base / self._dataset).glob(str(path / "*.parquet")):
            from contextlib import suppress

            with suppress(ValueError):
                # warn about this in the future
                frames.append(pl.scan_parquet(f, low_memory=low_memory))

        # Is this actually desired?
        if len(frames) == 0:
            return pl.LazyFrame()

        # https://github.com/pola-rs/polars/issues/12508
        return pl.concat(frames, how=how)

    def write(
        self,
        base: Path,
        frame: pl.LazyFrame,
        **kwargs: typing.Any,
    ) -> None:
        if len(set(kwargs.keys()) - set(self._keys)) > 0:
            msg = f"Got extra partition keys {"', '".join(kwargs.keys())}"
            raise KeyMismatchError(msg)

        if len(self._keys) > len(kwargs):
            values_df = (
                (frame.clone().filter(**kwargs) if len(kwargs) > 0 else frame)
                .select(self._keys)
                .unique()
                .collect(streaming=True)
            )

            parts = values_df.to_dicts()
            del values_df
            gc.collect()
        else:
            parts = [kwargs]

        batch = str(uuid4())
        for part in parts:
            path = Path(*[f"{k}={part[k]}" for k in self._keys])

            (base / self._dataset / path).mkdir(mode=0o755, parents=True, exist_ok=True)
            partition = frame.clone()
            partition.filter(**part).drop(self._keys).sink_parquet(
                base / self._dataset / path / f"{batch}-0.parquet",
                maintain_order=False,
            )
            del partition
            gc.collect()
