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
# %conda install polars=="0.19.13" --yes

# %%
import polars as pl

if hasattr(__builtins__, "__IPYTHON__"):
    from rich import inspect, print

    inspect("ruff/isort strip out rich")
    print("unless it is used")

    pl.show_versions()


# %%
def combine(adversaries: pl.LazyFrame, combos: pl.LazyFrame) -> pl.LazyFrame:
    """Combines spirits and adversaries to create the complete set of games."""

    all_combos = (
        combos.clone().drop("Hash", "Complexity").rename({"NComplexity": "Complexity"})
    )

    return (
        adversaries.clone()
        .join(all_combos, on="Matchup")
        .with_columns(
            [
                pl.col("Difficulty").add(pl.col("Difficulty_right")),
                pl.col("Complexity").add(pl.col("Complexity_right")),
            ]
        )
        .drop("Difficulty_right", "Complexity_right", "Matchup")
        .with_columns(pl.col("Difficulty").clip(lower_bound=0))
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
    max_difficulty = mean + 2 * stddev

    stats = (
        all_games.clone()
        .select(pl.col("Difficulty"), pl.col("Complexity"))
        .filter(pl.col("Difficulty").gt(0) & pl.col("Difficulty").lt(max_difficulty))
        .with_columns(
            [
                pl.col("Difficulty")
                .qcut([0.25, 0.5, 0.75], labels=["1", "2", "3", "4"])
                .cast(pl.Utf8)
                .str.to_integer()
                .cast(pl.Int8)
                .alias("Difficulty Range"),
                pl.col("Complexity")
                .qcut(
                    [0.25, 0.5, 0.75],
                    labels=["0", "1", "2", "3"],
                )
                .cast(pl.Utf8)
                .str.to_integer()
                .cast(pl.Int8)
                .alias("Complexity Range"),
            ]
        )
    )

    # qcut isn't supported by sink_parquet as of 0.20.2
    return (
        stats.clone()
        .select("Difficulty", "Difficulty Range")
        .unique()
        .collect(streaming=True)
        .lazy(),
        stats.clone()
        .select("Complexity", "Complexity Range")
        .unique()
        .collect(streaming=True)
        .lazy(),
    )


def filter_by_bucket(
    bucket: tuple[int, int],
    difficulty: pl.LazyFrame,
    complexity: pl.LazyFrame,
    games: pl.LazyFrame,
) -> pl.LazyFrame:
    return (
        games.clone()
        .join(
            complexity,
            on="Complexity",
        )
        .join(
            difficulty,
            on="Difficulty",
            how="left",
        )
        .drop("Difficulty", "Complexity")
        .filter(
            (
                pl.col("Difficulty Range").is_null()
                if bucket[0] == 0
                else pl.col("Difficulty Range").eq(bucket[0])
            )
            & pl.col("Complexity Range").eq(bucket[1])
        )
        .drop("Difficulty Range", "Complexity Range")
    )


# %%
