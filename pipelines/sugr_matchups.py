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

# %% [markdown]
# #

# %%
# %conda install prefect pandas --yes

# %%
import pandas as pd
from rich import inspect

# %%
inspect("ruff")

# %%
complexity_values = {
    "Low": 0,
    "Moderate": 1,
    "High": 2,
    "Very High": 4,
}


def get_spirits_by_expansions(expansions: int) -> pd.DataFrame:
    spirits = pd.read_csv("data/spirits.tsv", delimiter="\t")
    spirits = spirits[(expansions | spirits.Expansions) == expansions]
    spirits.Aspect = spirits.Aspect.fillna("")
    agg = (
        spirits.groupby("Name")
        .agg(Aspects=("Aspect", "count"))
        .reset_index()
        .set_index("Name")
    )
    # inspect(agg)

    spirits = spirits.join(agg, on="Name")
    spirits.Aspect = spirits.apply(
        lambda r: "Base" if r.Aspect == "" and r.Aspects > 1 else r.Aspect, axis=1
    )
    spirits = spirits.drop(["Expansions", "Aspects"], axis=1)

    spirits.Complexity = spirits.apply(
        lambda r: complexity_values[r.Complexity], axis=1
    )
    return spirits


spirits = get_spirits_by_expansions(47)
# inspect(spirits)

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


def calculate_matchups(
    expansions: int, matchup: str, spirits: pd.DataFrame
) -> pd.DataFrame:
    spirits = spirits[(spirits[matchup] != "Unplayable")]

    spirits = spirits.assign(
        Difficulty=spirits.apply(lambda row: difficulty_values[row[matchup]], axis=1),
    )[["Name", "Difficulty", "Aspect", "Complexity"]]

    agg = (
        spirits.groupby("Name")
        .agg({"Difficulty": "min", "Complexity": "max"})
        .reset_index()
        .set_index("Name")
    )
    # inspect(agg)

    spirits = spirits.join(agg, on="Name", rsuffix="_agg")
    spirits = (
        spirits[spirits.Difficulty == spirits.Difficulty_agg]
        .drop(["Complexity", "Difficulty_agg"], axis=1)
        .rename(columns={"Complexity_agg": "Complexity"})
    )
    # Create Name + ([Aspect] Recommended)

    return spirits


matchups = calculate_matchups(0, "Sweden", spirits)
# inspect(matchups)

# %%
