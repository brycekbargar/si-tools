# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# ruff: noqa

# %%
# %conda install polars --yes

# %%
import typing

import polars as pl


# %%
def adversaries_by_expansions(
    expansions: int,
    adversaries: pl.LazyFrame,
    escalations: pl.LazyFrame,
) -> tuple[pl.LazyFrame, list[str]]:
    """Filter and clean Adversary data and Matchup data, padding if necessary with escalations."""
    adversaries = (
        adversaries.clone()
        .filter(pl.col("Expansion").or_(expansions).eq(expansions))
        .drop("Expansion")
        .with_columns(
            [
                pl.col("Difficulty").cast(pl.Int8),
                pl.col("Complexity").cast(pl.Int8),
                pl.col("Level").cast(pl.Int8),
            ],
        )
        .rename({"Name": "Adversary"})
    )

    if adversaries.clone().select(pl.col("Adversary").n_unique()).collect().item() <= 3:
        adversaries = pl.concat(
            [
                adversaries,
                (
                    escalations.filter(pl.col("Expansion").and_(expansions).eq(0))
                    .drop("Expansion")
                    .with_columns(
                        [
                            pl.col("Difficulty").cast(pl.Int8),
                            pl.col("Complexity").cast(pl.Int8),
                        ],
                    )
                ),
            ],
            how="diagonal",
        )

    matchups = [
        typing.cast(str, m[0])
        for m in adversaries.clone()
        .unique(subset=["Matchup"])
        .select(pl.col("Matchup"))
        .collect(streaming=True)
        .rows()
    ]

    return (adversaries, matchups)


# %%
if hasattr(__builtins__, "__IPYTHON__"):
    (all_adversaries, matchups) = adversaries_by_expansions(
        63,
        pl.scan_csv("../data/adversaries.tsv", separator="\t"),
        pl.scan_csv("../data/escalations.tsv", separator="\t"),
    )

# %%
