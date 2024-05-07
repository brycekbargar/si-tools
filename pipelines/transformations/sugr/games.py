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

# ruff: noqa

# %%
# %conda install polars --yes

# %%
import polars as pl


# %%
def combine(adversaries: pl.LazyFrame, combos: pl.LazyFrame) -> pl.LazyFrame:
    """Combines spirits and adversaries to create the complete set of games."""
    all_combos = (
        combos.clone()
        .drop("Hash", "Difficulty", "Complexity")
        .rename({"NDifficulty": "Difficulty", "NComplexity": "Complexity"})
    )

    return (
        adversaries.clone()
        .join(all_combos, on="Matchup")
        .with_columns(
            [
                pl.col("Difficulty").add(pl.col("Difficulty_right")),
                pl.col("Complexity").add(pl.col("Complexity_right")),
            ],
        )
        .drop("Difficulty_right", "Complexity_right", "Matchup")
    )


# %%
if hasattr(__builtins__, "__IPYTHON__"):
    # TODO: Redo harness
    pass


# %%
def define_buckets(all_games: pl.LazyFrame) -> tuple[pl.LazyFrame, pl.LazyFrame]:
    """Find difficulty/complexity ranges to bucket games into."""
    (mean, stddev) = (
        all_games.clone()
        .select(
            pl.col("Difficulty").mean().alias("Difficulty Mean"),
            pl.col("Difficulty").std().alias("Difficulty Std"),
        )
        .collect(streaming=True)
        .row(0)
    )
    min_difficulty = mean - 2 * stddev
    max_difficulty = mean + 2 * stddev

    stats = (
        all_games.clone()
        .select(pl.col("Difficulty"), pl.col("Complexity"))
        .filter(
            pl.col("Difficulty").gt(min_difficulty)
            & pl.col("Difficulty").lt(max_difficulty),
        )
        .with_columns(
            [
                pl.col("Difficulty")
                .qcut(5, labels=["0", "1", "2", "3", "4"])
                .cast(pl.Utf8)
                .str.to_integer()
                .cast(pl.Int8)
                .alias("Difficulty Bucket"),
                pl.col("Complexity")
                .qcut(5, labels=["0", "1", "2", "3", "4"])
                .cast(pl.Utf8)
                .str.to_integer()
                .cast(pl.Int8)
                .alias("Complexity Bucket"),
            ],
        )
    )

    # qcut isn't supported by sink_parquet as of 0.20.2
    return (
        stats.clone()
        .group_by("Difficulty Bucket")
        .agg(
            pl.min("Difficulty").name.suffix(" Min"),
            pl.max("Difficulty").name.suffix(" Max"),
        )
        .rename({"Difficulty Bucket": "Bucket"})
        .collect(streaming=True)
        .lazy(),
        stats.clone()
        .group_by("Complexity Bucket")
        .agg(
            pl.min("Complexity").name.suffix(" Min"),
            pl.max("Complexity").name.suffix(" Max"),
        )
        .rename({"Complexity Bucket": "Bucket"})
        .collect(streaming=True)
        .lazy(),
    )


# %%
if hasattr(__builtins__, "__IPYTHON__"):
    # TODO: Redo harness
    buckets = define_buckets(
        pl.scan_parquet("../data/temp/1704403407472060/*_games.parquet"),
    )


# %%
def filter_by_bucket(
    bucket: tuple[int, int],
    difficulty: pl.LazyFrame,
    complexity: pl.LazyFrame,
    games: pl.LazyFrame,
) -> pl.LazyFrame:
    """Filter the given set of games to bucket based on difficulty/complexity."""
    (diff_min, diff_max) = (
        difficulty.clone()
        .filter(pl.col("Bucket").eq(bucket[0]))
        .select("Difficulty Min", "Difficulty Max")
        .collect(streaming=True)
        .row(0)
    )
    (comp_min, comp_max) = (
        complexity.clone()
        .filter(pl.col("Bucket").eq(bucket[1]))
        .select("Complexity Min", "Complexity Max")
        .collect(streaming=True)
        .row(0)
    )

    return games.clone().filter(
        pl.col("Difficulty").ge(diff_min)
        & pl.col("Difficulty").le(diff_max)
        & pl.col("Complexity").ge(comp_min)
        & pl.col("Complexity").le(comp_max),
    )


# %%
if hasattr(__builtins__, "__IPYTHON__"):
    # TODO: Redo harness
    pass
