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
        .cast({"Matchup": _all_matchups})
        .join(
            combos.clone().cast({"Matchup": _all_matchups}),
            on=["Expansion", "Matchup"],
        )
        .with_columns(
            pl.col("Difficulty").mul(pl.col("Difficulty_right")),
            pl.col("Complexity").add(pl.col("Complexity_right")),
        )
        .drop("Matchup")
    )


@dataclass
class Bucket:
    """Represents the parameters necessary to bucket non-horizons games."""

    name: str
    expr: pl.Expr
    difficulty: int
    complexity: int

    def __str__(self) -> str:
        return f"{self.name} ({self.difficulty}, {self.complexity})"


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
        pl.Expr.and_(
            sp_filter,
            pl.col("Players").gt(pl.lit(1)),
            pl.col("Players").le(pl.lit(4)),
        ),
    )

    difficulty = (
        all_games.clone()
        .with_columns(
            pl.col("Difficulty")
            .qcut(
                difficulty_count,
                labels=[str(label) for label in range(difficulty_count)],
                include_breaks=True,
            )
            .alias("qcut"),
        )
        .unnest("qcut")
        .select("breakpoint", "category")
        .unique()
    ).collect(streaming=True)

    d_min = -99
    for d_max, d in difficulty.sort("category").rows():
        if complexity_count == 0:
            yield Bucket(
                name,
                pl.Expr.and_(
                    sp_filter,
                    pl.col("Difficulty").gt(d_min),
                    pl.col("Difficulty").le(d_max),
                ),
                d,
                0,
            )
            continue

        complexity = (
            all_games.clone()
            .filter(
                pl.col("Difficulty").gt(d_min),
                pl.col("Difficulty").le(d_max),
            )
            .with_columns(
                pl.col("Complexity")
                .qcut(
                    complexity_count,
                    labels=[str(label) for label in range(complexity_count)],
                    include_breaks=True,
                )
                .alias("qcut"),
            )
            .unnest("qcut")
            .select("breakpoint", "category")
            .unique()
        ).collect(streaming=True)

        c_min = -99
        for c_max, c in complexity.sort("category").rows():
            yield Bucket(
                name,
                pl.Expr.and_(
                    sp_filter,
                    pl.col("Difficulty").gt(d_min),
                    pl.col("Difficulty").le(d_max),
                    pl.col("Complexity").gt(c_min),
                    pl.col("Complexity").le(c_max),
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
    nonbird = _buckets(
        "Pre Jagged Earth (No Bird)",
        all_games,
        pl.Expr.and_(
            pl.col("Expansion").lt(pl.lit(17)),
            pl.col("Expansion").ne(pl.lit(2)),
            pl.col("Spirit_0").ne_missing("Finder of Paths Unseen"),
            pl.col("Spirit_1").ne_missing("Finder of Paths Unseen"),
            pl.col("Spirit_2").ne_missing("Finder of Paths Unseen"),
            pl.col("Spirit_3").ne_missing("Finder of Paths Unseen"),
        ),
        difficulty_count=3,
        complexity_count=0,
    )

    bird = [
        Bucket(b.name, b.expr, b.difficulty, 1)
        for b in _buckets(
            "Pre Jagged Earth (Bird Only)",
            all_games,
            pl.Expr.and_(
                pl.col("Expansion").lt(pl.lit(49)),
                pl.col("Expansion").ne(pl.lit(2)),
                pl.Expr.or_(
                    pl.col("Spirit_0").eq_missing("Finder of Paths Unseen"),
                    pl.col("Spirit_1").eq_missing("Finder of Paths Unseen"),
                    pl.col("Spirit_2").eq_missing("Finder of Paths Unseen"),
                    pl.col("Spirit_3").eq_missing("Finder of Paths Unseen"),
                ),
            ),
            difficulty_count=3,
            complexity_count=0,
        )
    ]

    return [*nonbird, *bird]


def je_buckets(
    all_games: pl.LazyFrame,
) -> list[Bucket]:
    """Find difficulty/complexity ranges to bucket games into."""
    return list(
        _buckets(
            "Jagged Earth+",
            all_games,
            pl.Expr.and_(
                pl.col("Expansion").ge(pl.lit(17)),
                # Too many good games for D matchups.
                pl.Expr.not_(pl.col("Has D")),
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
        "Difficulty",
        "Complexity",
        "Has D",
    )


_all_matchups: pl.DataType = pl.Enum(
    [
        "Tier",
        "France",
        "Sweden",
        "Scotland",
        "Prussia",
        "Livestock",
        "England",
        "Russia",
        "Mines",
    ],
)
