"""Provides operations on LazyFrames finalizing Spirit Island games."""

import typing
from dataclasses import dataclass

import polars as pl


def create_games(
    adversaries: pl.LazyFrame,
    combos: pl.LazyFrame,
    *,
    use_expansion: bool = True,
) -> pl.LazyFrame:
    """Creates games from adversaries/spirits based on matchups."""
    return (
        adversaries.clone()
        .cast({"Matchup": _all_matchups})
        .join(
            combos.clone().cast({"Matchup": _all_matchups}),
            on=["Expansion", "Matchup"] if use_expansion else "Matchup",
        )
        .with_columns(
            pl.col("Difficulty").mul(pl.col("Difficulty_right")),
            pl.col("Complexity").mul(pl.lit(1.2)).add(pl.col("Complexity_right")),
        )
        .drop("Matchup", "Difficulty_right", "Complexity_right")
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


def preje_buckets(
    all_games: pl.LazyFrame,
) -> list[Bucket]:
    """Find difficulty/complexity ranges to bucket pre-jagged earth games into."""
    representative_games = all_games.clone().filter(
        pl.col("Expansion").eq(pl.lit(15)),
        pl.col("Players").gt(pl.lit(1)),
        pl.col("Players").le(pl.lit(3)),
    )

    difficulty = (
        representative_games.clone()
        .with_columns(
            pl.col("Difficulty")
            .qcut(
                3,
                labels=[str(label) for label in range(3)],
                include_breaks=True,
            )
            .alias("qcut"),
        )
        .unnest("qcut")
        .select("breakpoint", "category")
        .unique()
    ).collect(streaming=True)

    def _buckets() -> typing.Iterator[Bucket]:
        d_min = -99
        for d_max, d in difficulty.sort("category").rows():
            yield Bucket(
                "Pre Jagged Earth (No Birb)",
                pl.Expr.and_(
                    pl.col("Difficulty").gt(d_min),
                    pl.col("Difficulty").le(d_max),
                    pl.col("Spirit_0").ne_missing("Finder of Paths Unseen"),
                    pl.col("Spirit_1").ne_missing("Finder of Paths Unseen"),
                    pl.col("Spirit_2").ne_missing("Finder of Paths Unseen"),
                    pl.col("Spirit_3").ne_missing("Finder of Paths Unseen"),
                ),
                int(d),
                0,
            )
            yield Bucket(
                "Pre Jagged Earth (Birb)",
                pl.Expr.and_(
                    pl.col("Difficulty").gt(d_min),
                    pl.col("Difficulty").le(d_max),
                    pl.Expr.or_(
                        pl.col("Spirit_0").eq_missing("Finder of Paths Unseen"),
                        pl.col("Spirit_1").eq_missing("Finder of Paths Unseen"),
                        pl.col("Spirit_2").eq_missing("Finder of Paths Unseen"),
                        pl.col("Spirit_3").eq_missing("Finder of Paths Unseen"),
                    ),
                ),
                int(d),
                1,
            )
            d_min = d_max

    return list(_buckets())


def je_buckets(
    all_games: pl.LazyFrame,
) -> list[Bucket]:
    """Find difficulty/complexity ranges to bucket games into."""
    representative_games = all_games.clone().filter(
        pl.col("Expansion").eq(pl.lit(63)),
        # Too many good games for D matchups.
        pl.Expr.not_(pl.col("Has D")),
        pl.col("Players").gt(pl.lit(1)),
        pl.col("Players").le(pl.lit(4)),
    )

    (difficulty, complexity) = pl.collect_all(
        [
            representative_games.clone()
            .with_columns(
                pl.col("Difficulty")
                .qcut(
                    5,
                    labels=[str(label) for label in range(5)],
                    include_breaks=True,
                )
                .alias("qcut"),
            )
            .unnest("qcut")
            .select("breakpoint", "category")
            .cast({"category": pl.UInt8})
            .unique(),
            representative_games.clone()
            .with_columns(
                pl.col("Complexity")
                .qcut(
                    5,
                    labels=[str(label) for label in range(5)],
                    include_breaks=True,
                )
                .alias("qcut"),
            )
            .unnest("qcut")
            .select("breakpoint", "category")
            .cast({"category": pl.UInt8})
            .unique(),
        ],
        streaming=True,
    )

    def _buckets() -> typing.Iterator[Bucket]:
        d_min = -99
        for d_max, d in difficulty.sort("category").rows():
            c_min = -99
            for c_max, c in complexity.sort("category").rows():
                # Buckets are 0:0, 1:(1 + 2 + 3), 2:4
                if c in [1, 2]:
                    continue

                yield Bucket(
                    "Jagged Earth+",
                    pl.Expr.and_(
                        pl.Expr.not_(pl.col("Has D")),
                        pl.col("Difficulty").gt(d_min),
                        pl.col("Difficulty").le(d_max),
                        pl.col("Complexity").gt(c_min),
                        pl.col("Complexity").le(c_max),
                    ),
                    d,
                    c - 2 if c >= 3 else 0,
                )
                c_min = c_max
            d_min = d_max

    return list(_buckets())


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
