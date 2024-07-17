# dataset write / read roundtrip

import tempfile
import typing
from pathlib import Path
from uuid import uuid4

import polars as pl
import pytest
from pytest_cases import parametrize_with_cases

from flows.utilities.hive_dataset import HiveDataset as uut
from flows.utilities.hive_dataset import KeyMismatchError


class WriteCases:
    frame = pl.LazyFrame(
        {
            "int": [1, 1, 5, 5, 5, 2, 4],
            "string": ["a", "a", "b", "d", "b", "a", "b"],
            "values": ["1a", "1a", "5b", "5d", "5b", "2a", "4b"],
        },
    )

    def case_single_key_none(  # noqa:ANN201
        self,
    ):
        return (
            {"int": pl.UInt32},
            {},
            [("int=1", 2), ("int=2", 1), ("int=4", 1), ("int=5", 3)],
            WriteCases.frame.schema,
        )

    def case_single_key_one(  # noqa:ANN201
        self,
    ):
        return ({"int": pl.UInt32}, {"int": 1}, [("int=1", 2)], WriteCases.frame.schema)

    def case_single_key_mismatch(  # noqa:ANN201
        self,
    ):
        return (
            {"int": pl.UInt32},
            {"string": 1},
            KeyMismatchError,
            WriteCases.frame.schema,
        )

    def case_multiple_key_all(  # noqa:ANN201
        self,
    ):
        return (
            {"int": pl.UInt32, "string": pl.String},
            {"int": 1, "string": "a"},
            [("int=1/string=a", 2)],
            WriteCases.frame.schema,
        )

    def case_multiple_key_one(  # noqa:ANN201
        self,
    ):
        return (
            {"int": pl.UInt32, "string": pl.String},
            {"int": 5},
            [("int=5/string=b", 2), ("int=5/string=d", 1)],
            WriteCases.frame.schema,
        )

    def case_multiple_key_none(  # noqa:ANN201
        self,
    ):
        return (
            {"int": pl.UInt32, "string": pl.String},
            {},
            [
                ("int=1/string=a", 2),
                ("int=5/string=b", 2),
                ("int=5/string=d", 1),
                ("int=2/string=a", 1),
                ("int=4/string=b", 1),
            ],
            WriteCases.frame.schema,
        )

    def case_multiple_key_mismatch(  # noqa:ANN201
        self,
    ):
        return (
            {"int": pl.UInt32, "string": pl.String},
            {"banana": 1},
            KeyMismatchError,
            WriteCases.frame.schema,
        )

    def case_contextual_key_all(  # noqa:ANN201
        self,
    ):
        return (
            {"banana": pl.String, "int": pl.UInt32, "string": pl.String},
            {"int": 1, "banana": "mushy", "string": "a"},
            [("banana=mushy/int=1/string=a", 2)],
            {**WriteCases.frame.schema, "banana": pl.String},
        )

    def case_contextual_key_one(  # noqa:ANN201
        self,
    ):
        return (
            {"banana": pl.String, "int": pl.UInt32},
            {"banana": "mushy"},
            [
                ("banana=mushy/int=1", 2),
                ("banana=mushy/int=2", 1),
                ("banana=mushy/int=4", 1),
                ("banana=mushy/int=5", 3),
            ],
            {**WriteCases.frame.schema, "banana": pl.String},
        )

    def case_contextual_key_none(  # noqa:ANN201
        self,
    ):
        return (
            {"banana": pl.String},
            {},
            KeyMismatchError,
            {**WriteCases.frame.schema, "banana": pl.String},
        )

    def case_contextual_key_mismatch(  # noqa:ANN201
        self,
    ):
        return (
            {"banana": pl.String, "string": pl.String},
            {"int": 1},
            KeyMismatchError,
            {**WriteCases.frame.schema, "banana": pl.String},
        )


@parametrize_with_cases("schema, part, results, results_schema", cases=WriteCases)
def test_write(
    schema: dict[str, pl.DataType],
    part: dict[str, typing.Any],
    results: list[tuple[str, int]] | type,
    results_schema: dict[str, pl.DataType],
) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        name = str(uuid4())
        dataset = uut(tmpdir, name, **schema)

        if isinstance(results, type):
            with pytest.raises(results):  # type: ignore[reportArgumentType]
                dataset.write(WriteCases.frame, **part)
            return

        dataset.write(WriteCases.frame, **part)

        for path, height in results:
            partition = Path(tmpdir) / name / path
            assert partition.exists()
            files = 0
            for file in partition.iterdir():
                assert file.suffix == ".parquet"

                files += 1
                values = pl.read_parquet(file)
                assert dict(values.schema) == results_schema
                assert values.height == height

            assert files == 1
