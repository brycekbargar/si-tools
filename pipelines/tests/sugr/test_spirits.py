import typing

import polars as pl
import pytest


def test_spirits_by_expansion() -> None:
    from transformations.sugr.spirits import spirits_by_expansions as uut

    spirits = pl.LazyFrame(
        {
            "Name": ["S1", "S2", "S2", "S3", "S3", "S4"],
            "Aspect": [None, None, "2A1", None, "3A1", None],
            "Expansions": [1, 1, 4, 1, 8, 16],
            "Complexity": ["Moderate"] * 6,
        },
    )

    results = uut(1 | 8, spirits).collect(streaming=True).to_dict(as_series=False)

    # filters by expansion
    assert "S4" not in results["Name"]
    # only makes base aspect for spirit with multiple aspects
    assert results["Aspect"][results["Name"].index("S1")] is None
    assert results["Aspect"][results["Name"].index("S2")] is None
    # makes base aspect when appicable
    assert sorted(
        [
            a
            for a in results["Aspect"]
            if results["Name"][results["Aspect"].index(a)] == "S3"
        ],
    ) == sorted(["Base", "3A1"])

    # converts complexity to something usable
    for c in results["Complexity"]:
        assert isinstance(c, int)
        assert c > 0


def test_calculate_matchups() -> None:
    from transformations.sugr.spirits import calculate_matchups as uut

    matchup = "neiroatra"
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
            matchup: [
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

    results = uut(matchup, spirits).collect(streaming=True).to_dict(as_series=False)

    for i in range(len(results["Spirit"])):
        assert results["Matchup"][i] == matchup

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


def test_generate_combinations() -> None:
    from transformations.sugr.spirits import generate_combinations as uut

    spirits = pl.LazyFrame(
        {
            "Spirit": ["S1", "S2", "S3", "S4", "S5"],
            "Matchup": ["M1", "M1", "M2", "M2", "M2"],
            "Difficulty": [2, 4, 1, 3, 5],
            "Complexity": [4, 8, 2, 6, 10],
        },
    )
    s = spirits.collect(streaming=True).to_dict(as_series=False)

    m1lf = uut(spirits)
    m1 = m1lf.collect(streaming=True).to_dict(as_series=False)
    assert sorted(m1["Spirit_0"]) == sorted(s["Spirit"])

    m2lf = uut(spirits, previous_combos=m1lf)
    m2 = m2lf.collect(streaming=True).to_dict(as_series=False)
    assert len(m2["Spirit_0"]) == 4

    s1_s2_uniq = True
    s3_s4_uniq = True
    s3_s5_uniq = True
    s4_s5_uniq = True
    for i in range(len(m2["Spirit_0"])):
        match typing.cast(list[str], sorted((m2["Spirit_0"][i], m2["Spirit_1"][i]))):
            case ["S1", "S2"]:
                assert s1_s2_uniq
                s1_s2_uniq = False
                assert m2["NDifficulty"][i] == 3
                assert m2["NComplexity"][i] == 6
            case ["S3", "S4"]:
                assert s3_s4_uniq
                s3_s4_uniq = False
                assert m2["NDifficulty"][i] == 2
                assert m2["NComplexity"][i] == 4
            case ["S3", "S5"]:
                assert s3_s5_uniq
                s3_s5_uniq = False
                assert m2["NDifficulty"][i] == 3
                assert m2["NComplexity"][i] == 6
            case ["S4", "S5"]:
                assert s4_s5_uniq
                s4_s5_uniq = False
                assert m2["NDifficulty"][i] == 4
                assert m2["NComplexity"][i] == 8
            case _ as unknown:
                pytest.fail(f"{unknown} was unexpected")

    m3lf = uut(spirits, previous_combos=m2lf)
    m3 = m3lf.collect(streaming=True).to_dict(as_series=False)
    assert len(m3["Spirit_0"]) == 1

    assert sorted((m3["Spirit_0"][0], m3["Spirit_1"][0], m3["Spirit_2"][0])) == sorted(
        ["S3", "S4", "S5"],
    )
    assert m3["NDifficulty"][0] == 3
    assert m3["NComplexity"][0] == 6
