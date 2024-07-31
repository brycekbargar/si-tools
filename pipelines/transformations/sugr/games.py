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
            pl.col("NDifficulty").mul(pl.col("Difficulty")),
            pl.col("NComplexity").add(pl.col("Difficulty")),
        )
        .drop("Difficulty", "Complexity", "Matchup")
    )


@dataclass
class Bucket:
    """Represents the parameters necessary to bucket non-horizons games."""

    name: str
    expr: pl.Expr
    difficulty: int
    complexity: int

    def __str__(self) -> str:
        return f"{self.name} ({self.difficulty}, {self.complexity}):\n{self.expr}"


def horizons_bucket() -> Bucket:
    """Hardcoded bucket for horizons games."""
    return Bucket("Horizons", pl.col("Expansion").eq(2), 0, 0)


def _buckets(
    name: str,
    all_games: pl.LazyFrame,
    sp_filter: pl.Expr,
    difficulty_count: int,
    complexity_count: int,
) -> typing.Iterator[Bucket]:
    all_games = all_games.clone().filter(
        pl.and_(
            sp_filter,
            pl.col("Players").gt(pl.lit(1)),
            pl.col("Players").le(pl.lit(4)),
        ),
    )

    (difficulty, complexity) = pl.collect_all(
        [
            all_games.clone()
            .with_columns(
                pl.col("NDifficulty")
                .qcut(
                    difficulty_count,
                    labels=[str(label) for label in range(difficulty_count)],
                    include_breaks=True,
                )
                .alias("qcut"),
            )
            .unnest("qcut")
            .select("breakpoint", "category")
            .unique(),
            all_games.clone()
            .with_columns(
                pl.col("NComplexity")
                .qcut(
                    complexity_count,
                    labels=[str(label) for label in range(complexity_count)],
                    include_breaks=True,
                )
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
                name,
                pl.and_(
                    sp_filter,
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


def preje_buckets(
    all_games: pl.LazyFrame,
) -> list[Bucket]:
    """Find difficulty/complexity ranges to bucket pre-jagged earth games into."""
    return list(
        _buckets(
            "Pre Jagged Earth",
            all_games,
            pl.and_(
                pl.col("Expansion").lt(49),
                pl.col("Expansion").ne(2),
            ),
            difficulty_count=3,
            complexity_count=2,
        ),
    )


def je_buckets(
    all_games: pl.LazyFrame,
) -> list[Bucket]:
    """Find difficulty/complexity ranges to bucket games into."""
    return list(
        _buckets(
            "Jagged Earth+",
            all_games,
            pl.Expr.and_(
                pl.col("Expansion").ge(49),
                # Too many good games for D matchups.
                pl.not_(pl.col("Has D")),
            ),
            difficulty_count=5,
            complexity_count=3,
        ),
    )


def filter_by_bucket(
    bucket: Bucket,
    all_games: pl.LazyFrame,
) -> pl.LazyFrame:
    """Filter the given set of games to bucket based on difficulty/complexity."""
    return all_games.filter(bucket.expr).drop(
        "NDifficulty",
        "NComplexity",
        "Has D",
    )
