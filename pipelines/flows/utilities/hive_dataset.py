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
        keys = set(self._keys)
        contextual_keys = set(kwargs.keys())

        extra_keys = contextual_keys - keys
        if len(extra_keys) > 0:
            msg = f"Got extra partition keys: {"', '".join(extra_keys)}"
            raise KeyMismatchError(msg)

        if how == "vertical":
            if len(kwargs) == 0:
                return pl.scan_parquet(
                    self._dataset_path,
                    low_memory=low_memory,
                    hive_schema=self._schema,
                )

            return (
                pl.scan_parquet(
                    self._dataset_path,
                    low_memory=low_memory,
                    hive_schema=self._schema,
                )
                .filter(**kwargs)
                .drop(kwargs.keys())
                .cache()
            )

        # https://github.com/pola-rs/polars/issues/12508
        return pl.concat(
            (
                pl.scan_parquet(
                    f,
                    low_memory=low_memory,
                    hive_schema=self._schema,
                    hive_partitioning=True,
                )
                for f in self._dataset_path.glob(
                    str(
                        Path(*[f"{k}={kwargs.get(k, '*')}" for k in self._schema])
                        / "*.parquet",
                    ),
                )
            ),
            how=how,
        ).drop(kwargs.keys())

    def write(
        self,
        frame: pl.LazyFrame,
        **kwargs: typing.Any,
    ) -> None:
        keys = set(self._keys)
        contextual_keys = set(kwargs.keys())

        extra_keys = contextual_keys - keys
        if len(extra_keys) > 0:
            msg = f"Got extra partition keys: {"', '".join(extra_keys)}"
            raise KeyMismatchError(msg)

        schema = frame.collect_schema()
        columns = set(schema.names())
        del schema
        gc.collect()

        overspecified_keys = contextual_keys.intersection(columns)
        if len(overspecified_keys) > 0:
            msg = f"Contextual keys were found in the frame: {"', '".join(overspecified_keys)}"  # noqa: E501
            raise KeyMismatchError(msg)

        underspecified_keys = [
            k for k in keys if k not in columns and k not in contextual_keys
        ]
        if len(underspecified_keys) > 0:
            msg = f"No way to find the values for keys: {"', '".join(underspecified_keys)}"  # noqa: E501
            raise KeyMismatchError(msg)

        frame_keys = keys - contextual_keys
        if len(frame_keys) > 0:
            values = frame.clone().select(frame_keys).unique().collect(streaming=True)
            parts = [({**kwargs, **p}, p) for p in values.to_dicts()]
            del values
            gc.collect()

            if len(parts) == 0:
                msg = f"No unique values were found for keys: {"', '".join(frame_keys)}"
                raise KeyMismatchError(msg)
        else:
            parts = [(kwargs, {})]

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
