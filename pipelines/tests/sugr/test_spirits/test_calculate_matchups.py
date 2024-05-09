import typing

import polars as pl
import pytest

from transformations.sugr.spirits import calculate_matchups as uut


def test_calculate_matchups() -> None:
    spirits = pl.LazyFrame(
        {
            "Name": ["S1", "S1", "S2", "S2", "S2", "S3", "S4", "S4", "S5", "S5"],
            "Aspect": [
                "SA1",
                "S1A2",
                "S2A1",
                "S2A2",
                "S2A3",
                None,
                "S4A1",
                "S4A2",
                "S5A1",
                "S5A2",
            ],
            "Complexity": [1, 2, 1, 2, 3, 0, 0, 0, 0, 0],
            "Matchup": [
                "Mid+",
                "Mid+",
                "Counters",
                "Bottom",
                "Counters",
                "Mid-",
                "Unplayable",
                "Bottom",
                "Unplayable",
                "Unplayable",
            ],
        },
    )

    results = uut("Matchup", spirits).collect(streaming=True).to_dict(as_series=False)

    for i in range(len(results["Spirit"])):
        match typing.cast(str, results["Spirit"][i]):
            case s if s.startswith("S1"):
                # Any is for when all the aspects have the same matchup
                assert s == "S1 (Any)"
                # Take the lowest complexity
                assert results["Complexity"][i] == 2
            case s if s.startswith("S2"):
                # Aspects with the same matchup are combined
                assert s in ("S2 (S2A1, S2A3)", "S2 (S2A3, S2A1)")
                assert results["Complexity"][i] == 3
            case s if s.startswith("S3"):
                # Only with multiple aspects does a spirit get parens
                assert s == "S3"
            case s if s.startswith("S4"):
                # Unplayable spirts are excluded
                assert s == "S4 (S4A2)"
            case s if s.startswith("S5"):
                # No really, they're excluded
                pytest.fail("S5 was only unplayable")
            case _ as s:
                pytest.fail(f"{s} isn't an expected spirit")
