"""Provides operations on LazyFrames related to expansions."""

import polars as pl


def expansions_and_players(
    expansions: pl.LazyFrame,
    *,
    subset: bool = False,
    max_players: int = 6,
) -> list[tuple[int, list[int]]]:
    """Collects the unique expansions and their player counts."""
    if subset:
        expansions = expansions.clone().filter(
            pl.col("Expansion").is_in([1, 2, 15, 19, 31, 49, 63]),
        )

    values: list[tuple[int, int]] = (
        expansions.select("Expansion", "Players").collect(streaming=True).rows()
    )

    return [(exp, list(range(1, min(p, max_players) + 1))) for (exp, p) in values]


def horizons(
    frame: pl.LazyFrame,
) -> pl.LazyFrame:
    """Filters the frame to only-Horizons."""
    return frame.clone().filter(pl.col("Expansion").eq(pl.lit(2)))


def preje(
    frame: pl.LazyFrame,
) -> pl.LazyFrame:
    """Filters the frame to expansions pre Jagged Earth (except only-Horizons)."""
    return frame.clone().filter(
        pl.col("Expansion").lt(pl.lit(17)),
        pl.col("Expansion").ne(pl.lit(2)),
    )


def jaggedearth(
    frame: pl.LazyFrame,
) -> pl.LazyFrame:
    """Filters the frame to expansions post Jagged Earth."""
    return frame.clone().filter(pl.col("Expansion").ge(pl.lit(17)))
