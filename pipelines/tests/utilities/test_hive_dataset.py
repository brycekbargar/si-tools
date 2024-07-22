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
    frame = pl.DataFrame(
        {
            "int": [1, 1, 5, 5, 5, 2, 4],
            "string": ["a", "a", "b", "d", "b", "a", "b"],
            "values": ["1a", "1a", "5b", "5d", "5b", "2a", "4b"],
        },
    )

    def case_single_key(  # noqa:ANN201
        self,
    ):
        return (
            {"int": pl.UInt32},
            {},
            [("int=1", 2), ("int=2", 1), ("int=4", 1), ("int=5", 3)],
            WriteCases.frame.schema,
        )

    def case_single_key_extra(  # noqa:ANN201
        self,
    ):
        return (
            {"int": pl.UInt32},
            {"banana": "mushy"},
            KeyMismatchError,
            WriteCases.frame.schema,
        )

    def case_single_key_underspecified(  # noqa:ANN201
        self,
    ):
        return (
            {"banana": pl.String},
            {},
            KeyMismatchError,
            None,
        )

    def case_single_key_overspecified(  # noqa:ANN201
        self,
    ):
        return (
            {"int": pl.UInt32},
            {"int": 1},
            KeyMismatchError,
            None,
        )

    def case_single_key_mismatch(  # noqa:ANN201
        self,
    ):
        return (
            {"int": pl.UInt32},
            {"string": 1},
            KeyMismatchError,
            WriteCases.frame.schema,
        )

    def case_multiple_keys(  # noqa:ANN201
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

    def case_contextual_key(  # noqa:ANN201
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


@parametrize_with_cases(
    "schema, contextual_values, expected, expected_schema",
    cases=WriteCases,
)
def test_write(
    schema: dict[str, pl.DataType],
    contextual_values: dict[str, typing.Any],
    expected: list[tuple[str, int]] | type,
    expected_schema: dict[str, pl.DataType],
) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        name = str(uuid4())
        dataset = uut(tmpdir, name, **schema)

        if isinstance(expected, type):
            with pytest.raises(expected):  # type: ignore[reportArgumentType]
                dataset.write(WriteCases.frame.lazy(), **contextual_values)
            return

        dataset.write(WriteCases.frame.lazy(), **contextual_values)

        for path, expected_height in expected:
            partition = Path(tmpdir) / name / path
            assert partition.exists()
            files = 0
            for file in partition.iterdir():
                assert file.suffix == ".parquet"

                files += 1
                results = pl.read_parquet(file, hive_partitioning=True)
                assert dict(results.schema) == expected_schema
                assert results.height == expected_height

            assert files == 1


class ReadCases:
    def case_single_key(  # noqa:ANN201
        self,
    ):
        return (
            {"int": pl.UInt8},
            {
                "int=1": pl.DataFrame(
                    {"string": ["a", "a", "b"], "values": ["1a", "1a", "1b"]},
                ),
                "int=2": pl.DataFrame(
                    {"string": ["a", "a", "b"], "values": ["2a", "2a", "2b"]},
                ),
            },
            {},
            6,
            {"int": pl.UInt8, "string": pl.String, "values": pl.String},
        )

    def case_multiple_keys(  # noqa:ANN201
        self,
    ):
        return (
            {"int": pl.UInt8, "string": pl.String},
            {
                "int=1/string=a": pl.DataFrame(
                    {"values": ["1a", "1a"]},
                ),
                "int=1/string=b": pl.DataFrame(
                    {"values": ["1b"]},
                ),
                "int=2/string=a": pl.DataFrame(
                    {"values": ["2a", "2a"]},
                ),
                "int=2/string=b": pl.DataFrame(
                    {"values": ["2b"]},
                ),
            },
            {},
            6,
            {"int": pl.UInt8, "string": pl.String, "values": pl.String},
        )

    def case_diagonal_frames(  # noqa:ANN201
        self,
    ):
        return (
            {"int": pl.UInt8},
            {
                "int=1": pl.DataFrame(
                    {
                        "string": ["a", "a", "b"],
                        "values": ["1a", "1a", "1b"],
                        "extras": [666, 666, 666],
                    },
                ),
                "int=2": pl.DataFrame(
                    {"string": ["a", "a", "b"], "values": ["2a", "2a", "2b"]},
                ),
            },
            {},
            6,
            {
                "int": pl.UInt8,
                "string": pl.String,
                "values": pl.String,
                "extras": pl.Int64,
            },
        )

    def case_contextual_values(  # noqa:ANN201
        self,
    ):
        return (
            {"int": pl.UInt8},
            {
                "int=1": pl.DataFrame(
                    {"string": ["a", "a", "b"], "values": ["1a", "1a", "1b"]},
                ),
                "int=2": pl.DataFrame(
                    {"string": ["a", "a", "b"], "values": ["2a", "2a", "2b"]},
                ),
            },
            {"int": 2},
            3,
            {"string": pl.String, "values": pl.String},
        )

    def case_diagonal_frames_contextual_values(  # noqa:ANN201
        self,
    ):
        return (
            {"int": pl.UInt8, "string": pl.String},
            {
                "int=1/string=a": pl.DataFrame(
                    {
                        "values": ["1a", "1a"],
                        "extras": [666, 666],
                    },
                ),
                "int=1/string=c": pl.DataFrame(
                    {"values": ["1c", "1c"]},
                ),
                "int=2/string=a": pl.DataFrame(
                    {"values": ["2a", "2a"]},
                ),
            },
            {"int": 1},
            4,
            {
                "string": pl.String,
                "values": pl.String,
                "extras": pl.Int64,
            },
        )


@parametrize_with_cases(
    "schema, partition, read_opts, expected, expected_schema",
    cases=ReadCases,
)
def test_read(
    schema: dict[str, pl.DataType],
    partition: dict[str, pl.DataFrame],
    read_opts: dict[str, typing.Any],
    expected: int,
    expected_schema: dict[str, pl.DataType],
) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        name = str(uuid4())
        dataset = uut(tmpdir, name, **schema)

        for hive, frame in partition.items():
            path = Path(tmpdir, name, hive)
            path.mkdir(
                mode=0o755,
                parents=True,
                exist_ok=True,
            )
            frame.write_parquet(path / f"{uuid4()}.parquet")

        results = dataset.read(**read_opts).collect(streaming=True)
        assert dict(results.schema) == expected_schema
        assert results.height == expected
