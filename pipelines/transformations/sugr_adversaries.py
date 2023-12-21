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
if hasattr(__builtins__, "__IPYTHON__"):
    import polars as pl
    from rich import inspect, print

    inspect("ruff/isort strip out rich")
    print("unless it is used")


# %%
def adversaries_by_expansions(
    expansions: int, adversaries: pl.LazyFrame
) -> pl.LazyFrame:
    """Filter and clean Adversary data."""
    import polars as pl

    # Filter to just the given expansions.
    # Expansions is a bitfield, this is assuming that a superset of the expansions for an Adversary are required.
    adversaries = adversaries.filter(
        pl.col("Expansion").or_(expansions).eq(expansions)
    ).drop("Expansion")

    return adversaries


# %%
if hasattr(__builtins__, "__IPYTHON__"):
    all_adversaries = adversaries_by_expansions(
        31, pl.scan_csv("../data/adversaries.tsv", separator="\t")
    )

# %%
