import gc
import typing
from pathlib import Path
from uuid import uuid4

import polars as pl


class KeyMismatchError(Exception):
    pass


class HiveDataset:
    def __init__(self, base: str | Path, dataset: str, **kwargs: pl.DataType) -> None:
        self._dataset_path = Path(base) / dataset
        if len(kwargs) == 0:
            msg = "No partition keys provided"
            raise KeyMismatchError(msg)
        self._schema = kwargs

    def read(
        self,
        low_memory: bool = False,  # noqa: FBT001, FBT002
        how: typing.Literal["vertical", "diagonal"] = "vertical",
        **kwargs: typing.Any,
    ) -> pl.LazyFrame:
        path = Path(*[f"{k}={kwargs.get(k, '*')}" for k in self._schema])

        frames = []
        for f in self._dataset_path.glob(str(path / "*.parquet")):
            from contextlib import suppress

            with suppress(ValueError):
                # warn about this in the future
                frames.append(
                    pl.scan_parquet(f, low_memory=low_memory, hive_schema=self._schema),
                )

        # Is this actually desired?
        if len(frames) == 0:
            return pl.LazyFrame()

        # https://github.com/pola-rs/polars/issues/12508
        return pl.concat(frames, how=how)

    def write(
        self,
        frame: pl.LazyFrame,
        **kwargs: typing.Any,
    ) -> None:
        if len(set(kwargs.keys()) - set(self._schema.keys())) > 0:
            msg = f"Got extra partition keys {"', '".join(kwargs.keys())}"
            raise KeyMismatchError(msg)

        if len(self._schema) > len(kwargs):
            values_df = (
                (frame.clone().filter(**kwargs) if len(kwargs) > 0 else frame)
                .select(self._schema.keys())
                .unique()
                .collect(streaming=True)
            )

            parts = values_df.to_dicts()
            del values_df
            gc.collect()

            if len(parts) == 0:
                msg = "No unique values were found to partition by"
                raise KeyMismatchError(msg)
        else:
            parts = [kwargs]

        batch = str(uuid4())
        for part in parts:
            path = Path(
                *[f"{k}={part[k]}" for k in self._schema],
            )

            (self._dataset_path / path).mkdir(mode=0o755, parents=True, exist_ok=True)
            partition = frame.clone()
            partition.filter(**part).drop(self._schema.keys()).sink_parquet(
                self._dataset_path / path / f"{batch}-0.parquet",
                maintain_order=False,
            )
            del partition
            gc.collect()
