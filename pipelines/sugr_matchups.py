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
# %conda install prefect pandas --yes

# %%
import pandas as pd
from rich import inspect, print
from itertools import combinations

# %%
inspect("ruff")
print(pd.__version__)

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


matchups = calculate_matchups(0, "Sweden", spirits)
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


combos = gen_spirit_combinations(2, matchups)
# inspect(combos)

# %%
