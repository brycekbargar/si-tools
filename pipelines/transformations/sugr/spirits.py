"""Provides operations on LazyFrames related to spirits."""

import polars as pl


def spirits_by_expansions(expansions: int, spirits: pl.LazyFrame) -> pl.LazyFrame:
    """Filter, and clean Spirit data."""
    spirits = (
        spirits.clone()
        .filter(pl.col("Expansions").or_(expansions).eq_missing(expansions))
        .drop("Expansions", "Aspect")
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
                "Has D": [False] * 7,
            },
            schema={
                "Tier": pl.Utf8,
                "Difficulty": pl.Float32,
                "Has D": pl.Boolean,
            },
        )
    else:
        matchup_values = pl.LazyFrame(
            {
                matchup: ["S", "A", "B", "C", "D"],
                "Difficulty": [0.8, 0.9, 0.0, 1.15, 1.3],
                "Has D": [False, False, False, False, True],
            },
            schema={
                matchup: pl.Utf8,
                "Difficulty": pl.Float32,
                "Has D": pl.Boolean,
            },
        )

    return (
        (
            spirits.clone()
            .join(matchup_values, on=matchup)
            .group_by(["Spirit", "Difficulty"])
            .agg(pl.min("Complexity"), pl.all("Has D"))
            .filter(pl.col("Difficulty").eq(pl.min("Difficulty").over("Spirit")))
            .with_columns(pl.col("Spirit").hash().alias("Hash"))
        )
        .collect(streaming=True)
        .lazy()
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
                    pl.col("Difficulty").alias("NDifficulty"),
                    pl.col("Complexity").cast(pl.Float32).alias("NComplexity"),
                ],
            )
            .rename({"Spirit": "Spirit_0"})
            .select(
                pl.col("NDifficulty"),
                pl.col("Difficulty"),
                pl.col("NComplexity"),
                pl.col("Complexity"),
                pl.col("Has D"),
                pl.col("Spirit_0"),
                pl.col("Hash"),
            )
        )

    previous_schema = previous_combos.collect_schema()
    players = len([c for c in previous_schema.names() if c.startswith("Spirit")]) + 1
    sp_col = f"Spirit_{(players-1)}"
    spirit_n = pl.col(sp_col)
    unique_spirits = pl.Expr.not_(spirit_n.eq_missing(pl.col("Spirit_0")))
    cast_enum: dict[str, pl.DataType] = {"Spirit_0": _all_spirits}
    cast_string: dict[str, pl.DataType] = {"Spirit_0": pl.String}
    for i in range(players - 2, 0, -1):
        unique_spirits = unique_spirits.and_(
            pl.Expr.not_(spirit_n.eq_missing(pl.col(f"Spirit_{i}"))),
        )
        cast_enum[f"Spirit_{i+1}"] = _all_spirits
        cast_string[f"Spirit_{i+1}"] = pl.String

    return (
        matchups.clone()
        .join(
            previous_combos.clone(),
            how="cross",
        )
        .rename({"Spirit": sp_col})
        .cast(cast_enum)
        .filter(unique_spirits)
        .with_columns(
            pl.col("Difficulty").add(pl.col("Difficulty_right")),
            pl.col("Complexity").add(pl.col("Complexity_right")),
            pl.col("Has D").or_(pl.col("Has D_right")),
            pl.col("Hash").add(pl.col("Hash_right")),
        )
        .with_columns(
            pl.col("Difficulty").truediv(players).cast(pl.Float32).alias("NDifficulty"),
            pl.col("Complexity").truediv(players).cast(pl.Float32).alias("NComplexity"),
        )
        .drop("Difficulty_right", "Complexity_right", "Has D_right", "Hash_right")
        .cast(cast_string)
        .unique("Hash")
    )


_all_spirits = pl.Enum(
    [
        "Lightning's Swift Strike",
        "River Surges in Sunlight",
        "Vital Strength of the Earth",
        "Shadows Flicker Like Flame",
        "Thunderspeaker",
        "A Spread of Rampant Green",
        "Bringer of Dreams and Nightmares",
        "Ocean's Hungry Grasp",
        "Sharp Fangs Behind the Leaves",
        "Keeper of the Forbidden Wilds",
        "Devouring Teeth Lurk Underfoot",
        "Eyes Watch from the Trees",
        "Fathomless Mud of the Swamp",
        "Rising Heat of Stone and Sand",
        "Sun-Bright Whirlwind",
        "Stone's Unyielding Defiance",
        "Grinning Trickster Stirs Up Trouble",
        "Many Minds Move as One",
        "Volcano Looming High",
        "Shifting Memory of Ages",
        "Lure of the Deep Wilderness",
        "Vengeance as a Burning Plague",
        "Shroud of Silent Mist",
        "Fractured Days Split the Sky",
        "Starlight Seeks Its Form",
        "Heart of the Wildfire",
        "Serpent Slumbering Beneath the Island",
        "Downpour Drenches the World",
        "Finder of Paths Unseen",
        "Ember-Eyed Behemoth",
        "Hearth-Vigil",
        "Towering Roots of the Jungle",
        "Wandering Voice Keens Delirium",
        "Relentless Gaze of the Sun",
        "Wounded Waters Bleeding",
        "Breath of Darkness Down Your Spine",
        "Dances Up Earthquakes",
    ],
)
