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
def spirits_by_expansions(expansions: int) -> pl.LazyFrame:
    """Load, filter, and clean Spirit data from disk."""
    import polars as pl

    # Filter to just the given expansions.
    # Expansions is a bitfield, this is assuming that a superset of the expansions for a spirit are required.
    spirits = (
        pl.scan_csv("data/spirits.tsv", separator="\t")
        .filter(pl.col("Expansions").or_(expansions).eq(expansions))
        .drop("Expansions")
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
            .otherwise(pl.lit(""))
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
spirits_by_expansions(63).collect()

# %%
difficulty_values = {
    "Top": -1,
    "Counters": -1,
    "Mid+": 0,
    "Neutral": 0,
    "Mid-": 1,
    "Bottom": 2,
    "Unfavored": 2,
}


def calculate_matchups(matchup: str, spirits: pd.DataFrame) -> pd.DataFrame:
    matchups = spirits[(spirits[matchup] != "Unplayable")]

    matchups = matchups.assign(
        Difficulty=matchups.apply(lambda row: difficulty_values[row[matchup]], axis=1),
    )[["Name", "Difficulty", "Aspect", "Complexity"]]

    agg = (
        matchups.groupby("Name")
        .agg({"Difficulty": "min", "Complexity": "max"})
        .reset_index()
        .set_index("Name")
    )
    # inspect(agg)

    matchups = matchups.join(agg, on="Name", rsuffix="_agg")
    matchups = (
        matchups[matchups.Difficulty == matchups.Difficulty_agg]
        .drop(["Complexity", "Difficulty_agg"], axis=1)
        .rename(columns={"Complexity_agg": "Complexity"})
    )

    matchups = (
        matchups.groupby(["Name", "Difficulty", "Complexity"])
        .agg(Aspects=("Aspect", lambda g: [a for a in g if a != ""]))
        .reset_index()
    )
    matchups = matchups.assign(
        Spirit=matchups.apply(
            lambda r: r.Name
            + ("" if len(r.Aspects) == 0 else f" ({', '.join(r.Aspects)})"),
            axis=1,
        )
    )[["Spirit", "Difficulty", "Complexity"]]
    matchups.Spirit = matchups.Spirit.astype(pd.StringDtype())

    return matchups


# matchups = calculate_matchups("Sweden", spirits)
# inspect(matchups)
# matchups.dtypes


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
