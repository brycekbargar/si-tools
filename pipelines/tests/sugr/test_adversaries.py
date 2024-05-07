import polars as pl
import pytest

import transformations.sugr.adversaries as uut


@pytest.fixture()
def data() -> tuple[pl.LazyFrame, pl.LazyFrame]:
    return (
        pl.LazyFrame(
            {
                "Name": [
                    "A1",
                    "A1",
                    "A2",
                    "The Kingdom of France (Plantation Colony)",
                    "The Kingdom of France (Plantation Colony)",
                    "The Kingdom of France (Plantation Colony)",
                    "A4",
                    "A5",
                ],
                "Expansion": [1, 1, 1, 4, 4, 4, 8, 16],
                "Level": [1, 2, 1, 4, 5, 6, 1, 1],
                "Matchup": [
                    "One",
                    "One",
                    "Two",
                    "France",
                    "France",
                    "France",
                    "Four",
                    "Five",
                ],
                "Difficulty": [0] * 8,
                "Complexity": [0] * 8,
            },
        ),
        pl.LazyFrame(
            {
                "Escalation": ["E1", "E2", "E3", "E4"],
                "Expansion": [1, 4, 8, 16],
                "Difficulty": [0] * 4,
                "Complexity": [0] * 4,
            },
        ),
    )


def test_filters_and_collects(data: tuple[pl.LazyFrame, pl.LazyFrame]) -> None:
    (filtered, collected) = uut.adversaries_by_expansions(
        1 | 4 | 16,
        data[0],
        pl.LazyFrame(),
    )

    expected = data[0].collect().to_dict(as_series=False)
    filtered = (filtered.collect(streaming=True).to_dict(as_series=False))["Adversary"]

    assert sorted(filtered) == sorted([n for n in expected["Name"] if n != "A4"])
    assert sorted(collected) == sorted(
        {n for n in expected["Matchup"] if n != "Four"},
    )


def test_adds_escalations(data: tuple[pl.LazyFrame, pl.LazyFrame]) -> None:
    (escalations, _) = uut.adversaries_by_expansions(
        1 | 16,
        data[0],
        data[1],
    )

    (no_escalations, _) = uut.adversaries_by_expansions(
        1 | 4 | 8 | 16,
        data[0],
        data[1],
    )

    assert sorted(
        {
            e
            for e in escalations.collect(streaming=True).to_dict(
                as_series=False,
            )["Escalation"]
            if e is not None
        },
        # Only "new" escalations based on what expansions are included
    ) == sorted(["E2", "E3"])

    assert "Escalation" not in no_escalations.collect(streaming=True).to_dict(
        as_series=False,
    )


def test_fuck_france(data: tuple[pl.LazyFrame, pl.LazyFrame]) -> None:
    (adversaries, _) = uut.adversaries_by_expansions(
        1 | 4 | 8 | 16,
        data[0],
        pl.LazyFrame(),
    )

    france = (
        adversaries.filter(pl.col("Matchup").eq(pl.lit("France")))
        .collect()
        .to_dict(as_series=False)
    )

    assert len(france["Adversary"]) == 1
    assert france["Level"][0] == 4
