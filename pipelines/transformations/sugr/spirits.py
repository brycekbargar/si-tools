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
                    # https://discord.com/channels/846580409050857493/846580409050857496/1162666833015488573
                    "Complexity": ["Intro", "Low", "Moderate", "High", "Very High"],
                    # Fractured and Finder should always be in the highest bucket.
                    "Value": [0, 1, 3, 5, 255],
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
    if matchup == "Tier":
        matchup_values = pl.LazyFrame(
            {
                "Tier": ["X", "S", "A", "B", "C", "D", "F"],
                "Difficulty": [0.7, 0.8, 0.9, 0.0, 1.1, 1.2, 1.3],
            },
            schema={
                "Tier": pl.Utf8,
                "Difficulty": pl.Float32,
            },
        )
    else:
        matchup_values = pl.LazyFrame(
            {
                matchup: ["S", "A", "B", "C", "D"],
                "Difficulty": [0.8, 0.9, 0.0, 1.15, 1.3],
            },
            schema={
                matchup: pl.Utf8,
                "Difficulty": pl.Float32,
            },
        )

    return (
        spirits.clone()
        .join(matchup_values, on=matchup)
        .group_by(["Spirit", "Difficulty"])
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
