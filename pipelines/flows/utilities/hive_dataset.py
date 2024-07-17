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
        self._keys = list(kwargs.keys())

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
                    pl.scan_parquet(
                        f,
                        low_memory=low_memory,
                        hive_schema=self._schema,
                        hive_partitioning=True,
                    ),
                )

        # Is this actually desired?
        if len(frames) == 0:
            return pl.LazyFrame()

        # https://github.com/pola-rs/polars/issues/12508
        return pl.concat(frames, how=how).drop(kwargs.keys())

    def write(
        self,
        frame: pl.LazyFrame,
        **kwargs: typing.Any,
    ) -> None:
        keys = set(self._keys)
        if len(set(kwargs.keys()) - keys) > 0:
            msg = f"Got extra partition keys {"', '".join(kwargs.keys())}"
            raise KeyMismatchError(msg)

        schema = frame.collect_schema()
        for k in self._keys:
            if k not in kwargs and k not in schema:
                msg = f"No way to find values of key '{k}'"
                raise KeyMismatchError(msg)

        global_values = {k: kwargs[k] for k in kwargs if k not in schema}
        filters = {k: kwargs[k] for k in kwargs if k in schema}
        del schema
        gc.collect()

        if len(self._schema) > len(kwargs):
            values_lf = frame.clone()
            values_df = (
                (
                    (values_lf.filter(**filters) if len(filters) > 0 else values_lf)
                    if len(filters) > 0
                    else frame
                )
                .select(keys - set(global_values.keys()))
                .unique()
                .collect(streaming=True)
            )

            parts = [({**global_values, **f}, f) for f in values_df.to_dicts()]
            del values_lf
            del values_df
            gc.collect()

            if len(parts) == 0:
                msg = "No unique values were found to partition by"
                raise KeyMismatchError(msg)
        else:
            parts = [({**global_values, **filters}, filters)]

        batch = str(uuid4())
        for segment, part in parts:
            path = Path(*[f"{k}={segment[k]}" for k in self._keys])

            (self._dataset_path / path).mkdir(mode=0o755, parents=True, exist_ok=True)
            partition = frame.clone()
            (
                partition.filter(**part).drop(part.keys())
                if len(part) > 0
                else partition
            ).sink_parquet(
                self._dataset_path / path / f"{batch}-0.parquet",
                maintain_order=False,
            )
            del partition
            gc.collect()
