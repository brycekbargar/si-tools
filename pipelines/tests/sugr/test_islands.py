import polars as pl
import pytest

import transformations.sugr.islands as uut


@pytest.fixture()
def data() -> tuple[pl.LazyFrame, pl.LazyFrame]:
    return (
        pl.LazyFrame(
            {
                "Layout": ["1p1", "1p2", "2p", "3p1", "3p2", "3p3"],
                "Players": [1, 1, 2, 3, 3, 3],
                "Weight": [50, 150, 100, 50, 50, 100],
                "Standard": [False, True] + [False] * 3 + [True],
            },
        ),
        pl.LazyFrame(
            {
                "Boards": ["1b1", "3b", "2b", "1b2"],
                "Players": [1, 3, 2, 1],
            },
        ),
    )


def test_explode_layouts(data: tuple[pl.LazyFrame, pl.LazyFrame]) -> None:
    layouts = (
        uut.explode_layouts(data[0], 3).collect(streaming=True).to_dict(as_series=False)
    )

    assert 47 <= sum([lout == "3p3" for lout in layouts["Layout"]]) <= 53
    assert 22 <= sum([lout == "3p1" for lout in layouts["Layout"]]) <= 28
    assert 22 <= sum([lout == "3p2" for lout in layouts["Layout"]]) <= 38

    assert sorted(layouts.keys()) == sorted(["Layout", "Standard"])


def test_generate_loose_islands(data: tuple[pl.LazyFrame, pl.LazyFrame]) -> None:
    islands = (
        uut.generate_loose_islands(
            data[0].filter(pl.col("Players") == 1),
            data[1].filter(pl.col("Players") == 1),
        )
        .collect(streaming=True)
        .to_dict(as_series=False)
    )

    expected = sorted(
        [
            ("1p1", "1b1"),
            ("1p1", "1b2"),
            ("1p2", "1b1"),
            ("1p2", "1b2"),
        ],
    )

    assert (
        sorted(
            [
                (islands["Layout"][i], islands["Boards"][i])
                for i in range(len(islands["Layout"]))
            ],
        )
        == expected
    )
    assert islands["Layout"][0] == "1p2"
