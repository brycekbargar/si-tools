"""Provides operations on LazyFrames related to adversaries."""

import typing

import polars as pl


def adversaries_by_expansions(
    expansions: int,
    adversaries: pl.LazyFrame,
    escalations: pl.LazyFrame,
) -> tuple[pl.LazyFrame, list[str]]:
    """Filter and clean Adversary data and Matchup data.

    With three or fewer adveraries the list will be padded with additional
    escalations based on adversaries not already included.

    With more than four adversaries, France level 5 & 6 will be removed.
    """
    adversaries = (
        adversaries.clone()
        .filter(pl.col("Expansion").or_(expansions).eq_missing(expansions))
        .drop("Expansion")
        .with_columns(
            [
                pl.col("Difficulty").cast(pl.UInt8),
                pl.col("Level").cast(pl.UInt8),
            ],
        )
        .cast(
            {
                "Name": pl.String,
                "Matchup": pl.String,
            },
        )
    )

    adv_count = adversaries.clone().unique("Name").collect().height

    if adv_count <= 3:
        adversaries = pl.concat(
            [
                adversaries,
                (
                    escalations.filter(
                        pl.col("Expansion").and_(expansions).eq_missing(0),
                    )
                    .drop("Expansion")
                    .with_columns(
                        [
                            pl.lit("Escalation").alias("Name"),
                            pl.lit(1).cast(pl.UInt8).alias("Difficulty"),
                            pl.lit("Intro").alias("Complexity"),
                            pl.lit("Tier").alias("Matchup"),
                        ],
                    )
                ),
            ],
            how="diagonal",
        )

    if adv_count > 4:
        adversaries = adversaries.filter(
            (
                pl.col("Name")
                .eq(pl.lit("The Kingdom of France (Plantation Colony)"))
                .not_()
            ).or_(pl.col("Level").le(pl.lit(4))),
        )

    matchups = [
        typing.cast(str, m[0])
        for m in adversaries.clone()
        .unique(subset=["Matchup"])
        .select(pl.col("Matchup"))
        .collect(streaming=True)
        .rows()
    ]

    return (
        adversaries.join(
            pl.LazyFrame(
                {
                    "Complexity": ["Intro", "Low", "Moderate", "High", "Very High"],
                    "Value": [0, 1, 3, 6, 127],
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
        .rename({"Name": "Adversary", "Value": "Complexity"}),
        matchups,
    )
