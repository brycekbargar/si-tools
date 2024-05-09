"""Provides operations on LazyFrames finalizing Spirit Island games."""

import polars as pl


def combine(adversaries: pl.LazyFrame, combos: pl.LazyFrame) -> pl.LazyFrame:
    """Combines spirits and adversaries to create the complete set of games."""
    return (
        adversaries.clone()
        .join(combos.clone(), on="Matchup")
        .with_columns(
            [
                pl.col("Difficulty").add(pl.col("Difficulty_right")),
                pl.col("Complexity").add(pl.col("Complexity_right")),
            ],
        )
        .drop("Difficulty_right", "Complexity_right", "Matchup")
    )


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
            pl.min("Difficulty").alias("Min"),
            pl.max("Difficulty").alias("Max"),
        )
        .rename({"Difficulty Bucket": "Bucket"})
        .collect(streaming=True)
        .lazy(),
        stats.clone()
        .group_by("Complexity Bucket")
        .agg(
            pl.min("Complexity").alias("Min"),
            pl.max("Complexity").alias("Max"),
        )
        .rename({"Complexity Bucket": "Bucket"})
        .collect(streaming=True)
        .lazy(),
    )


def filter_by_bucket(
    bucket: tuple[int, int],
    difficulty: pl.LazyFrame,
    complexity: pl.LazyFrame,
    games: pl.LazyFrame,
) -> pl.LazyFrame:
    """Filter the given set of games to bucket based on difficulty/complexity."""

    def min_max(of: pl.LazyFrame, bucket: int) -> tuple[int, int]:
        return (
            of.clone()
            .filter(pl.col("Bucket").eq(bucket))
            .select("Min", "Max")
            .collect(streaming=True)
            .row(0)
        )

    (diff_min, diff_max) = min_max(of=difficulty, bucket=bucket[0])
    (comp_min, comp_max) = min_max(of=complexity, bucket=bucket[1])

    return games.clone().filter(
        pl.col("Difficulty").ge(diff_min)
        & pl.col("Difficulty").le(diff_max)
        & pl.col("Complexity").ge(comp_min)
        & pl.col("Complexity").le(comp_max),
    )
