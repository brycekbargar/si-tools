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
scm = Step("SugrGamesFlow/1703370583255314/calculate_matchups")
# inspect(scm)
err_task = [t for t in scm.tasks() if t.successful is not True][0]
inspect(err_task.data)
adversaries = err_task.data.adversaries
spirits = err_task.data.spirits

# %%
from transformations.sugr_adversaries import unique_matchups

matchups = list([r[0] for r in unique_matchups(adversaries).collect().rows()])

# %%
from transformations.sugr_spirits import calculate_matchups

for m in matchups:
    print(calculate_matchups(m, spirits).collect())

# %%
