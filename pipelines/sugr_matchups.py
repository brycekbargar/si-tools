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
difficulty_values = {
    "Top": -1,
    "Counters": -1,
    "Mid+": 0,
    "Neutral": 0,
    "Mid-": 1,
    "Bottom": 2,
    "Unfavored": 2,
}
complexity_values = {
    "Low": 0,
    "Moderate": 1,
    "High": 2,
    "Very High": 4,
}


def generate_spirits(count: int, expansions: int, matchup: str) -> pd.DataFrame:
    spirits = pd.read_csv("data/spirits.tsv", delimiter="\t", dtype={"Aspect": str})
    spirits = spirits[
        ((expansions | spirits.Expansions) == expansions)
        & (spirits[matchup] != "Unplayable")
    ]
    spirits.Aspect = spirits.Aspect.fillna("")

    spirits = spirits.assign(
        Difficulty=spirits.apply(lambda row: difficulty_values[row[matchup]], axis=1),
        Complexity=spirits.apply(
            lambda row: complexity_values[row.PrintedComplexity], axis=1
        ),
    )[["Name", "Difficulty", "Aspect", "Complexity"]]

    agg = (
        spirits.groupby("Name")
        .agg({"Difficulty": "min", "Complexity": "max", "Aspect": "count"})
        .reset_index()
        .set_index("Name")
    )
    # inspect(agg)

    spirits = spirits.join(agg, on="Name", rsuffix="_agg")
    spirits = (
        spirits[spirits.Difficulty == spirits.Difficulty_agg]
        .assign(
            Aspect=spirits.apply(
                lambda row: row.Aspect
                if row.Aspect != ""
                else "Base"
                if row.Aspect_agg > 1
                else "",
                axis=1,
            ),
        )
        .drop(["Complexity", "Difficulty_agg", "Aspect_agg"], axis=1)
        .rename(columns={"Complexity_agg": "Complexity"})
    )

    return spirits
    # NaN -> Base if multiple aspects)
    # Drop extra columns
    # Group by Name, Difficulty -> [Aspects], Max(Complexity)
    # Create Name + ([Aspect] Recommended)


# %%
inspect(generate_spirits(1, 47, "Sweden"))

# %%

# %%
