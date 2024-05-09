import typing

import polars as pl
import pytest

from transformations.sugr.spirits import generate_combinations as uut


def test_generate_combinations() -> None:
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
