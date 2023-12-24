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
def combine(
    players: int, adversaries: pl.LazyFrame, matchups: list[pl.LazyFrame]
) -> pl.LazyFrame:
    """Combines spirits and adversaries to create the complete set of games."""
    import polars as pl

    from .sugr_spirits import generate_combinations

    return (
        adversaries.join(
            pl.concat([generate_combinations(players, m) for m in matchups]),
            on="Matchup",
        )
        .with_columns(
            [
                pl.col("Difficulty").add(pl.col("Difficulty_right")),
                pl.col("Complexity").add(pl.col("Complexity_right")),
            ]
        )
        .drop("Difficulty_right", "Complexity_right", "Matchup")
    )


# %%
if hasattr(__builtins__, "__IPYTHON__"):
    import sugr_spirits as ss
    import sugr_adversaries as sa
    import polars as pl

    spirits = ss.spirits_by_expansions(
        31, pl.scan_csv("../data/spirits.tsv", separator="\t")
    )
    adversaries = sa.adversaries_by_expansions(
        31, pl.scan_csv("../data/adversaries.tsv", separator="\t"), None
    )
    matchups = [
        ss.calculate_matchups(m[0], spirits)
        for m in sa.unique_matchups(adversaries).collect().rows()
    ]

    print(combine(3, adversaries, matchups).bottom_k(10, by=["Difficulty"]).collect())

# %%

# %%
