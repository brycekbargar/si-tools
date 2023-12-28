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
# %mamba install metaflow polars --yes

# %%
# ruff: noqa: E402
import random
import typing
from pathlib import Path

import polars as pl
from metaflow import Flow

# %%
difficulty = 2
complexity = 2

sugr_games = Flow("SugrGamesFlow")

temp = typing.cast(Path, sugr_games.latest_successful_run.data.temp)
total_games = (
    pl.read_csv(temp / "stats.tsv", separator="\t")
    .filter(
        (pl.col("Expansions") == 31)
        & (pl.col("Players") == 2)
        & (pl.col("Difficulty") == difficulty)
        & (pl.col("Complexity") == complexity)
    )
    .select("Count")
).item()
r_game = random.randrange(total_games)
print(r_game)

(
    pl.scan_parquet(temp / f"3102{difficulty:02}{complexity:02}.parquet")
    .head(r_game + 1)
    .last()
).collect().glimpse()

# %%

# %%
