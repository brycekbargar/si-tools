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
    assert "S4" not in results["Spirit"]

    # converts complexity to something usable
    for c in results["Complexity"]:
        assert isinstance(c, int)
        assert c > 0


def test_calculate_matchups() -> None:
    from transformations.sugr.spirits import calculate_matchups as uut

    matchup = "neiroatra"
    spirits = pl.LazyFrame(
        {
            "Spirit": ["S1", "S1", "S2", "S2", "S2", "S3", "S3", "S4", "S4"],
            "Complexity": [1, 2, 2, 1, 3, 0, 0, 2, 2],
            matchup: ["A", "A", "S", "D", "S", "F", "F", "D", "D"],
        },
    )

    results = uut(matchup, spirits).collect(streaming=True).to_dict(as_series=False)

    assert len(results["Spirit"]) == 3
    for i in range(len(results["Spirit"])):
        match typing.cast(str, results["Spirit"][i]):
            case s if s == "S1":
                # Take the lowest complexity
                assert results["Complexity"][i] == 1
                assert not results["Has D"][i]
            case s if s == "S2":
                # Take the lowest complexity among the best matchups
                assert results["Complexity"][i] == 2
                assert results["Difficulty"][i] == pytest.approx(0.8)
                assert not results["Has D"][i]
            case s if s == "S4":
                assert results["Has D"][i]
            case s if s == "S3":
                # F-tier matchups are excluded
                pytest.fail("S3 was only f-tier")
            case _ as s:
                pytest.fail(f"{s} isn't an expected spirit")


def test_generate_combinations() -> None:
    from transformations.sugr.spirits import generate_combinations as uut

    spirits = pl.LazyFrame(
        {
            "Spirit": ["S1", "S2", "S3"],
            "Difficulty": [2, 4, 3],
            "Complexity": [4, 8, 3],
            "Has D": [False, True, False],
        },
    )
    s = spirits.collect(streaming=True).to_dict(as_series=False)

    m1lf = uut(spirits)
    m1 = m1lf.collect(streaming=True).to_dict(as_series=False)
    assert sorted(m1["Spirit_0"]) == sorted(s["Spirit"])

    m2lf = uut(spirits, previous_combos=m1lf)
    m2 = m2lf.collect(streaming=True).to_dict(as_series=False)
    assert len(m2["Spirit_0"]) == 3

    s1_s2_uniq = True
    s1_s3_uniq = True
    s2_s3_uniq = True
    for i in range(len(m2["Spirit_0"])):
        match typing.cast(list[str], sorted((m2["Spirit_0"][i], m2["Spirit_1"][i]))):
            case ["S1", "S2"]:
                assert s1_s2_uniq
                s1_s2_uniq = False
                assert m2["NDifficulty"][i] == 3
                assert m2["NComplexity"][i] == 6
                assert m2["Has D"][i]
            case ["S1", "S3"]:
                assert s1_s3_uniq
                s1_s3_uniq = False
                assert m2["NDifficulty"][i] == 2.5
                assert m2["NComplexity"][i] == 3.5
                assert not m2["Has D"][i]
            case ["S2", "S3"]:
                assert s2_s3_uniq
                s2_s3_uniq = False
                assert m2["NDifficulty"][i] == 3.5
                assert m2["NComplexity"][i] == 5.5
                assert m2["Has D"][i]
            case _ as unknown:
                pytest.fail(f"{unknown} was unexpected")

    m3lf = uut(spirits, previous_combos=m2lf)
    m3 = m3lf.collect(streaming=True).to_dict(as_series=False)
    assert len(m3["Spirit_0"]) == 1

    assert sorted((m3["Spirit_0"][0], m3["Spirit_1"][0], m3["Spirit_2"][0])) == sorted(
        ["S1", "S2", "S3"],
    )
    assert m3["NDifficulty"][0] == 3
    assert m3["NComplexity"][0] == 5
    assert m3["Has D"][0]
