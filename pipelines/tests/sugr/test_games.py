import polars as pl
import pytest

import transformations.sugr.games as uut


def test_define_buckets() -> None:
    games: dict[str, list[float]] = {
        "Difficulty": [
            -2.21,
            -0.09,
            1.01,
            1.07,
            1.91,
            2.39,
            2.42,
            2.63,
            3.25,
            3.57,
            3.72,
            3.79,
            3.84,
            3.88,
            3.96,
            4.55,
            5.86,
            6.74,
            8.03,
            10.09,
        ],
        "Complexity": [
            # Two standard deviations for difficulty get cut
            # But not complexity so these are out of order to test
            4.46,
            3.23,
            3.9,
            3.93,
            4.03,
            4.46,
            4.69,
            4.78,
            4.8,
            5,
            5.44,
            6.36,
            6.36,
            6.51,
            6.59,
            7.92,
            8.42,
            9.7,
            9.75,
            7.84,
        ],
    }

    results = uut.define_buckets(pl.LazyFrame(games))

    (difficulty, complexity) = (
        r.sort("Bucket").collect(streaming=True).to_dict(as_series=False)
        for r in results
    )

    assert len(difficulty["Bucket"]) == 5
    assert len(complexity["Bucket"]) == 5

    last_d_max: float = 0
    last_c_max: float = 0
    for i in range(5):
        if i == 0:
            assert difficulty["Min"][i] == -0.09
            assert complexity["Min"][i] == 3.23

        if i > 0:
            assert difficulty["Min"][i] > last_d_max
            assert complexity["Min"][i] > last_c_max
        last_d_max = difficulty["Max"][i]
        last_c_max = complexity["Max"][i]

        if i == 4:
            assert difficulty["Max"][i] == 8.03
            assert complexity["Max"][i] == 9.75


@pytest.mark.parametrize("bucket", [(0, 0), (0, 1), (1, 0), (1, 1)])
def filter_by_bucket(bucket: tuple[int, int]) -> None:
    difficulty = pl.LazyFrame(
        {
            "Bucket": [0, 1],
            "Min": [1, 3],
            "Max": [2, 4],
        },
    )
    complexity = pl.LazyFrame(
        {
            "Bucket": [0, 1],
            "Min": [5, 7],
            "Max": [6, 8],
        },
    )
    games = pl.LazyFrame(
        {
            "Difficulty": [1, 2, 3, 4],
            "Complexity": [5, 6, 7, 8],
        },
    )

    results = (
        uut.filter_by_bucket(bucket, difficulty, complexity, games)
        .select(pl.len())
        .collect(streaming=True)
        .item()
    )

    assert results == 1
