import tempfile
from pathlib import Path
from uuid import uuid4

import polars as pl

from flows.utilities.dataset import Dataset as uut


def test_roundtrip() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)

        name = str(uuid4())
        write_dataset = uut(name)

        frame1 = pl.LazyFrame(
            {
                "int": [1, 1, 5, 5, 5, 2, 4],
                "string": ["a", "a", "b", "d", "b", "a", "b"],
                "values": ["1a", "1a", "5b", "5d", "5b", "2a", "4b"],
            },
        )
        frame2 = pl.LazyFrame(
            {
                "int": [1, 1, 5, 5, 5, 2, 4],
                "wing": ["a", "a", "b", "d", "b", "a", "b"],
                "values": ["1a", "1a", "5b", "5d", "5b", "2a", "4b"],
            },
        )

        write_dataset.write(base, frame1)
        write_dataset.write(base, frame2)

        read_dataset = uut(name)
        read_frame = read_dataset.read(base, how="diagonal")

        assert read_frame.schema == {
            "int": pl.Int64,
            "string": pl.String,
            "wing": pl.String,
            "values": pl.String,
        }

        assert read_frame.collect().height == 14
