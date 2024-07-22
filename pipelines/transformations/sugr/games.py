"""Provides operations on LazyFrames finalizing Spirit Island games."""

import gc

import polars as pl


def define_buckets(
    adversaries: pl.LazyFrame,
    combos: pl.LazyFrame,
) -> tuple[int, int, int, float, float, int, float, float]:
    """Find difficulty/complexity ranges to bucket games into."""
    all_games = (
        adversaries.clone()
        .join(combos.clone(), on=["Expansion", "Matchup"])
        .with_columns(
            [
                pl.col("Difficulty").add(pl.col("Difficulty_right")),
                pl.col("Complexity").add(pl.col("Complexity_right")).truediv(pl.lit(2)),
            ],
        )
        .drop("Difficulty_right", "Complexity_right", "Matchup")
    )

    distribution = all_games.clone().select(
        pl.col("Difficulty").mean().alias("Difficulty Mean"),
        pl.col("Difficulty").std().alias("Difficulty Std"),
        pl.col("Complexity").mean().alias("Complexity Mean"),
        pl.col("Complexity").std().alias("Complexity Std"),
    )
    (dmean, dstddev, cmean, cstddev) = distribution.collect(streaming=True).row(0)
    min_difficulty = dmean - 2 * dstddev
    max_difficulty = dmean + 2 * dstddev
    min_complexity = cmean - 2 * cstddev
    max_complexity = cmean + 2 * cstddev
    del distribution
    gc.collect()

    buckets = (
        all_games.clone()
        .filter(
            pl.col("Difficulty").gt(min_difficulty)
            & pl.col("Difficulty").lt(max_difficulty)
            & pl.col("Complexity").gt(min_complexity)
            & pl.col("Complexity").lt(max_complexity),
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
                .qcut(3, labels=["0", "1", "2"])
                .cast(pl.Utf8)
                .str.to_integer()
                .cast(pl.Int8)
                .alias("Complexity Bucket"),
            ],
        )
        .group_by("Expansion", "Players", "Difficulty Bucket", "Complexity Bucket")
        .agg(
            pl.min("Difficulty").alias("Min Difficulty"),
            pl.max("Difficulty").alias("Max Difficulty"),
            pl.min("Complexity").alias("Min Complexity"),
            pl.max("Complexity").alias("Max Complexity"),
        )
        .select(
            "Expansion",
            "Players",
            "Difficulty Bucket",
            "Min Difficulty",
            "Max Difficulty",
            "Complexity Bucket",
            "Min Complexity",
            "Max Complexity",
        )
    )

    collected_buckets = buckets.collect(streaming=True).rows()
    del all_games
    del buckets
    gc.collect()

    return collected_buckets  # type: ignore[reportReturnType]


def filter_by_bucket(
    difficulty: tuple[float, float],
    complexity: tuple[float, float],
    adversaries: pl.LazyFrame,
    combos: pl.LazyFrame,
) -> pl.LazyFrame:
    """Filter the given set of games to bucket based on difficulty/complexity."""
    return (
        adversaries.clone()
        .join(combos.clone(), on="Matchup")
        .with_columns(
            [
                pl.col("Difficulty").add(pl.col("Difficulty_right")),
                pl.col("Complexity").add(pl.col("Complexity_right")).truediv(pl.lit(2)),
            ],
        )
        .filter(
            pl.col("Difficulty").ge(difficulty[0])
            & pl.col("Difficulty").le(difficulty[1])
            & pl.col("Complexity").ge(complexity[0])
            & pl.col("Complexity").le(complexity[1]),
        )
        .drop(
            "Difficulty_right",
            "Difficulty",
            "NDifficulty",
            "Complexity_right",
            "Complexity",
            "NComplexity",
            "Hash",
            "Matchup",
        )
    )
