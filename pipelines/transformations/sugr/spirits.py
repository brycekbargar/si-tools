"""Provides operations on LazyFrames related to spirits."""

import polars as pl


def spirits_by_expansions(expansions: int, spirits: pl.LazyFrame) -> pl.LazyFrame:
    """Filter, and clean Spirit data."""
    all_spirits = (
        spirits.clone()
        .select("Name")
        .unique()
        .collect(streaming=True)
        .to_series()
        .to_list()
    )
    spirits = (
        spirits.clone()
        .filter(pl.col("Expansions").or_(expansions).eq(expansions))
        .drop("Expansions", "Aspect")
        .cast({"Name": pl.Enum(all_spirits)})
    )

    return (
        spirits.join(
            pl.LazyFrame(
                {
                    "Complexity": ["Low", "Moderate", "High", "Very High"],
                    "Value": [0, 1, 2, 4],
                },
                schema={
                    "Complexity": pl.String,
                    "Value": pl.UInt8,
                },
            ),
            on="Complexity",
            how="left",
        )
        .drop("Complexity")
        .rename({"Name": "Spirit", "Value": "Complexity"})
    )


def calculate_matchups(matchup: str, spirits: pl.LazyFrame) -> pl.LazyFrame:
    """Calculate the difficulty modifiers and best/worst spirits for the matchup."""
    spirit_matchups = (
        spirits.clone()
        .filter(pl.Expr.not_(pl.col(matchup).eq(pl.lit("U"))))
        .join(
            pl.LazyFrame(
                {
                    matchup: [
                        "X",
                        "S",
                        "A",
                        "B",
                        "C",
                        "D",
                        "F",
                    ],
                    "Difficulty": [-4, -2, -1, 0, 1, 2, 4],
                },
                schema={
                    matchup: pl.Utf8,
                    "Difficulty": pl.Int8,
                },
            ),
            on=matchup,
            how="left",
        )
    )

    return (
        spirit_matchups.group_by(["Spirit", "Difficulty"])
        .agg(pl.min("Complexity"))
        .filter(pl.col("Difficulty").eq(pl.min("Difficulty").over("Spirit")))
    )


def generate_combinations(
    matchups: pl.LazyFrame,
    previous_combos: pl.LazyFrame | None = None,
) -> pl.LazyFrame:
    """Generate all possible combinations of spirits."""
    if previous_combos is None:
        return (
            matchups.clone()
            .with_columns(
                [
                    pl.col("Difficulty").cast(pl.Float32).alias("NDifficulty"),
                    pl.col("Complexity").cast(pl.Float32).alias("NComplexity"),
                    pl.col("Spirit").hash().alias("Hash"),
                ],
            )
            .rename({"Spirit": "Spirit_0"})
            .select(
                pl.col("NDifficulty"),
                pl.col("Difficulty"),
                pl.col("NComplexity"),
                pl.col("Complexity"),
                pl.col("Spirit_0"),
                pl.col("Hash"),
            )
        )

    previous_schema = previous_combos.collect_schema()
    players = len([c for c in previous_schema.names() if c.startswith("Spirit")]) + 1
    sp_col = f"Spirit_{(players-1)}"
    spirit_n = pl.col(sp_col)
    unique_spirits = pl.Expr.not_(spirit_n.eq(pl.col("Spirit_0")))
    for i in range(players - 2, 0, -1):
        unique_spirits = unique_spirits.and_(
            pl.Expr.not_(spirit_n.eq(pl.col(f"Spirit_{i}"))),
        )

    return (
        matchups.clone()
        .join(
            previous_combos.clone(),
            how="cross",
        )
        .rename({"Spirit": sp_col})
        .filter(unique_spirits)
        .with_columns(
            pl.col("Difficulty").add(pl.col("Difficulty_right")),
            pl.col("Complexity").add(pl.col("Complexity_right")),
            pl.col("Hash").add(pl.col(sp_col).hash()),
        )
        .with_columns(
            pl.col("Difficulty").truediv(players).cast(pl.Float32).alias("NDifficulty"),
            pl.col("Complexity").truediv(players).cast(pl.Float32).alias("NComplexity"),
        )
        .drop("Difficulty_right", "Complexity_right")
        .unique("Hash")
    )
