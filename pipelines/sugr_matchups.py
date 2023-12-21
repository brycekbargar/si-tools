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

# %%
# %conda install polars pandas --yes

# %%
from itertools import combinations

import pandas as pd

if hasattr(__builtins__, "__IPYTHON__"):
    import polars as pl
    from rich import inspect, print

    inspect("ruff/isort strip out rich")
    print("unless it is used")


# %%
def spirits_by_expansions(expansions: int, spirits: pl.LazyFrame) -> pl.LazyFrame:
    """Filter, and clean Spirit data."""
    import polars as pl

    # Filter to just the given expansions.
    # Expansions is a bitfield, this is assuming that a superset of the expansions for a spirit are required.
    spirits = spirits.filter(pl.col("Expansions").or_(expansions).eq(expansions)).drop(
        "Expansions"
    )

    # Cleanup the Aspect column.
    # We only care to label spirits as the "Base" version when aspects are available.
    spirits = (
        spirits.join(spirits.group_by("Name").count(), on="Name", how="left")
        .with_columns(
            pl.when(pl.col("Aspect").is_not_null())
            .then(pl.col("Aspect"))
            .when(pl.col("count").gt(1))
            .then(pl.lit("Base"))
            .otherwise(None)
            .alias("Aspect")
        )
        .drop("count")
    )

    # Convert Complexity from text to numeric.
    spirits = (
        spirits.join(
            pl.LazyFrame(
                {
                    "Complexity": ["Low", "Moderate", "High", "Very High"],
                    "Value": [0, 1, 2, 4],
                }
            ),
            on="Complexity",
            how="left",
        )
        .with_columns(pl.col("Value").alias("Complexity"))
        .drop("Value")
    )

    return spirits


# %%
all_spirits = spirits_by_expansions(63, pl.scan_csv("data/spirits.tsv", separator="\t"))


# %%
def calculate_matchups(matchup: str, spirits: pl.LazyFrame) -> pl.LazyFrame:
    """Calculate the difficulty modifiers and best spirits for the matchup."""
    import polars as pl

    # Convert matchhup from text to numeric difficulty.
    matchups = (
        spirits.filter(pl.Expr.not_(pl.col(matchup).eq(pl.lit("Unplayable"))))
        .join(
            pl.LazyFrame(
                {
                    matchup: [
                        "Top",
                        "Counters",
                        "Mid+",
                        "Neutral",
                        "Mid-",
                        "Bottom",
                        "Unfavored",
                    ],
                    "Difficulty": [-1, -1, 0, 0, 1, 2, 2],
                }
            ),
            on=matchup,
            how="left",
        )
        .select(
            pl.col("Name"),
            pl.col("Difficulty"),
            pl.col("Aspect"),
            pl.col("Complexity"),
        )
    )

    # Find the best aspects.
    matchups = (
        matchups.group_by(["Name", "Difficulty"])
        .agg([pl.col("Aspect"), pl.max("Complexity")])
        .filter(pl.col("Difficulty").eq(pl.min("Difficulty").over("Name")))
        .with_columns(
            pl.when(pl.col("Aspect").list.drop_nulls().list.len().eq(0))
            .then(pl.col("Name"))
            .otherwise(
                pl.concat_str(
                    [
                        pl.col("Name"),
                        pl.lit(" ("),
                        pl.col("Aspect").list.join(", "),
                        pl.lit(")"),
                    ]
                )
            )
            .alias("Spirit")
        )
    )

    # Reorder and clean up columns.
    matchups = matchups.with_columns(pl.lit(matchup).alias("Matchup")).select(
        pl.col("Matchup"), pl.col("Difficulty"), pl.col("Complexity"), pl.col("Spirit")
    )

    return matchups


# %%
matchups = calculate_matchups("England", all_spirits)


# %%
def _spirit_labels(count: int) -> list[tuple[str, str]]:
    return combinations(["Spirit"] + [f"Spirit_{s+1}" for s in range(count)], 2)


def gen_spirit_combinations(count: int, spirits: pd.DataFrame) -> pd.DataFrame:
    combos = spirits[["Difficulty", "Complexity", "Spirit"]]
    for i in range(1, count):
        combos = combos.merge(spirits, how="cross", suffixes=(None, f"_{i}"))
        combos = combos.assign(
            Difficulty=combos.apply(
                lambda r: r.Difficulty + r[f"Difficulty_{i}"], axis=1
            ),
            Complexity=combos.apply(
                lambda r: r.Complexity + r[f"Complexity_{i}"], axis=1
            ),
        ).drop([f"Difficulty_{i}", f"Complexity_{i}"], axis=1)
        for p in _spirit_labels(i):
            combos = combos[combos[p[0]] != combos[p[1]]]

    combos = combos.rename(columns={"Spirit": "Spirit_0"})
    return combos


# combos = gen_spirit_combinations(2, matchups)
# inspect(combos)

# %%
