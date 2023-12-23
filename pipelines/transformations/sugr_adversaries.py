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
# %conda install polars --yes

# %%
import polars as pl

if hasattr(__builtins__, "__IPYTHON__"):
    from rich import inspect, print

    inspect("ruff/isort strip out rich")
    print("unless it is used")


# %%
def adversaries_by_expansions(
    expansions: int, adversaries: pl.LazyFrame, escalations: pl.LazyFrame
) -> pl.LazyFrame:
    """Filter and clean Adversary data, padding if necessary."""
    import polars as pl

    # Filter to just the given expansions.
    # Expansions is a bitfield, this is assuming that a superset of the expansions for an Adversary are required.
    adversaries = adversaries.filter(
        pl.col("Expansion").or_(expansions).eq(expansions)
    ).drop("Expansion")

    if adversaries.clone().select(pl.col("Name").n_unique()).collect().item() > 3:
        return adversaries

    # Filter out the given expansions.
    escalations = escalations.filter(pl.col("Expansion").and_(expansions).eq(0)).drop(
        "Expansion"
    )

    return pl.concat([adversaries, escalations], how="diagonal")


# %%
if hasattr(__builtins__, "__IPYTHON__"):
    all_adversaries = adversaries_by_expansions(
        2,
        pl.scan_csv("../data/adversaries.tsv", separator="\t"),
        pl.scan_csv("../data/escalations.tsv", separator="\t"),
    )

# %%
