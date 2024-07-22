"""Provides operations on LazyFrames finalizing Spirit Island games."""

import gc
import typing

import polars as pl


def create_games(
    adversaries: pl.LazyFrame,
    combos: pl.LazyFrame,
) -> pl.LazyFrame:
    """Creates games from adversaries/spirits based on matchups, filtering outliers."""
    all_games = (
        adversaries.clone()
        .join(
            combos.clone().drop("Difficulty", "Complexity", "Hash"),
            on=["Expansion", "Matchup"],
        )
        .with_columns(
            pl.col("NDifficulty").add(pl.col("Difficulty")),
            pl.col("NComplexity").add(pl.col("Difficulty")).truediv(pl.lit(2)),
        )
        .drop("Difficulty", "Complexity", "Matchup")
    )

    distribution = all_games.clone().select(
        pl.col("NDifficulty").mean().alias("DStd"),
        pl.col("NDifficulty").std().alias("DMean"),
        pl.col("NComplexity").mean().alias("CStd"),
        pl.col("NComplexity").std().alias("CMean"),
    )
    (dmean, dstddev, cmean, cstddev) = distribution.collect(streaming=True).row(0)
    min_difficulty = dmean - 2 * dstddev
    max_difficulty = dmean + 2 * dstddev
    min_complexity = cmean - 2 * cstddev
    max_complexity = cmean + 2 * cstddev
    del distribution
    gc.collect()

    # sink_parquet doesn't support streaming filtering by std/mean as of 1.1
    return all_games.filter(
        pl.col("NDifficulty").ge(pl.lit(min_difficulty))
        & pl.col("NDifficulty").le(pl.lit(max_difficulty))
        & pl.col("NComplexity").ge(pl.lit(min_complexity))
        & pl.col("NComplexity").le(pl.lit(max_complexity)),
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
