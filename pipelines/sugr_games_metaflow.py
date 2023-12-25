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
# %conda install metaflow polars --yes

# %%
# ruff: noqa: E402
from metaflow import Step
from rich import print, inspect

inspect("ruff/isort strip out rich")
print("unless it is used")

# %%
scm = Step("SugrGamesFlow/1703373792723178/fanout_players")
# inspect(scm)
scm_task = [t for t in scm.tasks() if t.successful is True][0]
inspect(scm_task.data)
inspect(scm_task.artifacts)
adversaries = scm_task.data.adversaries
spirits = scm_task.data.spirits
matchups = scm_task.data.matchups

# %%
matchups[0].collect()

# %%
from transformations.sugr_spirits import generate_combinations
import polars as pl

combinations = adversaries.join(
    pl.concat([generate_combinations(2, m).collect().lazy for m in matchups]),
    on="Matchup",
)


# %%
combinations.show_graph(comm_subplan_elim=False, streaming=True)
combinations.sink_parquet("test.parquet")

# %%
import polars as pl

diff = pl.col("Difficulty")
comp = pl.col("Complexity")

all_games = pl.scan_parquet("./data/results/1703458137835502/*.parquet")
print(all_games.clone().select(diff, comp).collect(streaming=True).describe())
print(all_games.clone().head().collect())

# %%
import random

diff = (8, 9)
comp = (3, 8)

source_games = pl.scan_parquet("./data/results/1703458137835502/31 2.parquet")
filtered = source_games.filter(
    (pl.col("Difficulty") >= diff[0])
    & (pl.col("Difficulty") < diff[1])
    & (pl.col("Complexity") >= comp[0])
    & (pl.col("Complexity") < comp[1])
)
total = filtered.select(pl.count("Name")).collect().item()
print(filtered.collect(streaming=True).row(random.randrange(total)))

# %%

# %%
