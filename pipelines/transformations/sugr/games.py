"""Provides operations on LazyFrames finalizing Spirit Island games."""

import typing

import polars as pl


def create_games(
    adversaries: pl.LazyFrame,
    combos: pl.LazyFrame,
) -> pl.LazyFrame:
    """Creates games from adversaries/spirits based on matchups."""
    return (
        adversaries.clone()
        .join(
            combos.clone().drop("Difficulty", "Complexity", "Hash"),
            on=["Expansion", "Matchup"],
        )
        .with_columns(
            pl.col("NDifficulty").add(pl.col("Difficulty")),
            pl.col("NComplexity").add(pl.col("Difficulty")),
        )
        .drop("Difficulty", "Complexity", "Matchup")
    )


def define_buckets(
    all_games: pl.LazyFrame,
) -> typing.Iterator[tuple[int, float, float, int, float, float]]:
    """Find difficulty/complexity ranges to bucket games into."""
    # sink_parquet doesn't support streaming qcut as of 1.1
    (difficulty, complexity) = pl.collect_all(
        [
            all_games.clone()
            .with_columns(
                pl.col("NDifficulty")
                .qcut(5, labels=["0", "1", "2", "3", "4"], include_breaks=True)
                .alias("qcut"),
            )
            .unnest("qcut")
            .select("breakpoint", "category")
            .unique(),
            all_games.clone()
            .with_columns(
                pl.col("NComplexity")
                .qcut(3, labels=["0", "1", "2"], include_breaks=True)
                .alias("qcut"),
            )
            .unnest("qcut")
            .select("breakpoint", "category")
            .unique(),
        ],
        streaming=True,
    )

    d_min = -99
    for d_max, d in difficulty.sort("category").rows():
        c_min = -99
        for c_max, c in complexity.sort("category").rows():
            yield (d, d_min, d_max, c, c_min, c_max)
            c_min = c_max
        d_min = d_max


def filter_by_bucket(
    difficulty: tuple[float, float],
    complexity: tuple[float, float],
    all_games: pl.LazyFrame,
) -> pl.LazyFrame:
    """Filter the given set of games to bucket based on difficulty/complexity."""
    return all_games.filter(
        pl.col("NDifficulty").gt(difficulty[0])
        & pl.col("NDifficulty").le(difficulty[1])
        & pl.col("NComplexity").gt(complexity[0])
        & pl.col("NComplexity").le(complexity[1]),
    ).drop(
        "NDifficulty",
        "NComplexity",
    )
