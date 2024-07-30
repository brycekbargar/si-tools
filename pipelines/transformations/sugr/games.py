"""Provides operations on LazyFrames finalizing Spirit Island games."""

import typing
from dataclasses import dataclass

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


@dataclass
class Bucket:
    """Represents the parameters necessary to bucket non-horizons games."""

    expr: pl.Expr
    difficulty: int
    complexity: str


def horizons_bucket() -> typing.Iterator[Bucket]:
    """Hardcoded bucket for horizons games."""
    yield Bucket(pl.col("Expansion").eq(2), 0, "horizons")


def preje_buckets(
    all_games: pl.LazyFrame,
) -> typing.Iterator[Bucket]:
    """Find difficulty/complexity ranges to bucket pre-jagged earth games into."""
    preje = pl.and_(pl.col("Expansion").lt(49), pl.col("Expansion").ne(2))

    difficulty = (
        all_games.clone()
        .filter(pl.and_(preje, pl.col("Players").gt(pl.lit(1))))
        .with_columns(
            pl.col("NDifficulty")
            .qcut(3, labels=["0", "1", "2"], include_breaks=True)
            .alias("qcut"),
        )
        .unnest("qcut")
        .select("breakpoint", "category")
        .unique()
        .collect(streaming=True)
    )

    d_min = -99
    for d_max, d in difficulty.sort("category").rows():
        yield Bucket(
            pl.and_(
                preje,
                pl.col("NDifficulty").gt(d_min),
                pl.col("NDifficulty").le(d_max),
            ),
            d,
            "birb",
        )
        yield Bucket(
            pl.and_(
                preje,
                pl.col("NDifficulty").gt(d_min),
                pl.col("NDifficulty").le(d_max),
                pl.col("Spirit_0").ne("Finder of Paths Unseen"),
                pl.col("Spirit_1").ne("Finder of Paths Unseen"),
                pl.col("Spirit_2").ne("Finder of Paths Unseen"),
                pl.col("Spirit_3").ne("Finder of Paths Unseen"),
            ),
            d,
            "nobirb",
        )
        d_min = d_max


def je_buckets(
    all_games: pl.LazyFrame,
) -> typing.Iterator[Bucket]:
    """Find difficulty/complexity ranges to bucket games into."""
    je = pl.col("Expansion").ge(49)
    (difficulty, complexity) = pl.collect_all(
        [
            all_games.clone()
            .filter(
                pl.and_(
                    je,
                    pl.col("Players").gt(pl.lit(1)),
                    pl.col("Players").le(pl.lit(4)),
                ),
            )
            .with_columns(
                pl.col("NDifficulty")
                .qcut(5, labels=["0", "1", "2", "3", "4"], include_breaks=True)
                .alias("qcut"),
            )
            .unnest("qcut")
            .select("breakpoint", "category")
            .unique(),
            all_games.clone()
            .filter(
                pl.and_(
                    je,
                    pl.col("Players").gt(pl.lit(1)),
                    pl.col("Players").le(pl.lit(4)),
                ),
            )
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
            yield Bucket(
                pl.and_(
                    je,
                    pl.col("NDifficulty").gt(d_min),
                    pl.col("NDifficulty").le(d_max),
                    pl.col("NComplexity").gt(c_min),
                    pl.col("NComplexity").le(c_max),
                ),
                d,
                c,
            )
            c_min = c_max
        d_min = d_max


def filter_by_bucket(
    bucket: Bucket,
    all_games: pl.LazyFrame,
) -> pl.LazyFrame:
    """Filter the given set of games to bucket based on difficulty/complexity."""
    return all_games.filter(bucket.expr).drop(
        "NDifficulty",
        "NComplexity",
    )
