"""Provides operations on LazyFrames related to spirits."""

import polars as pl
import polars.selectors as cs


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
            .sort("Spirit", "Difficulty")
            .unique("Spirit", keep="first")
        )
        # something isn't support by sink_parquet as of 1.2.1
        .collect(streaming=True)
        .lazy()
    )


def generate_combinations(
    players: int,
    matchups: pl.LazyFrame,
    combos: pl.LazyFrame,
) -> pl.LazyFrame:
    """Filters and calculates complexity/difficulty for combinations of spirits."""
    if players == 1:
        return (
            matchups.clone()
            .with_columns(pl.col("Complexity").cast(pl.Float32))
            .rename({"Spirit": "Spirit_0"})
        )

    combos = combos.clone().cast({f"Spirit_{p}": _all_spirits for p in range(players)})
    matchups = matchups.clone().cast({"Spirit": _all_spirits})
    for p in range(players):
        combos = combos.clone().join(
            matchups,
            left_on=f"Spirit_{p}",
            right_on="Spirit",
            suffix=f"-{p}",
        )

    return (
        (
            combos.clone()
            .with_columns(
                pl.mean_horizontal(cs.starts_with("Difficulty")).alias("Difficulty"),
                pl.mean_horizontal(cs.starts_with("Complexity")).alias("Complexity"),
                pl.sum_horizontal(cs.starts_with("Has D"))
                .cast(pl.Boolean)
                .alias("Has D"),
            )
            .cast({f"Spirit_{p}": pl.String for p in range(players)})
            .select(
                [
                    *[f"Spirit_{p}" for p in range(players)],
                    "Difficulty",
                    "Complexity",
                    "Has D",
                ],
            )
        )
        # horizontal isn't support by sink_parquet as of 1.2.1
        .collect(streaming=True)
        .lazy()
    )


def _write_combinations(output: str) -> None:
    import csv
    from itertools import combinations
    from pathlib import Path

    for i in range(1, 7):
        combos_csv = Path(output, f"{i}.csv")
        combos_csv.unlink(missing_ok=True)
        with combos_csv.open("w+", newline="") as combos_file:
            writer = csv.writer(combos_file)
            writer.writerow([f"Spirit_{p}" for p in range(i)])

            for c in combinations(_all_spirits.categories.to_list(), i):
                writer.writerow(c)

        combos_parquet = Path(output, f"{i}.parquet")
        combos_parquet.unlink(missing_ok=True)
        pl.scan_csv(combos_csv).sink_parquet(combos_parquet)

        combos_csv.unlink(missing_ok=True)


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

if __name__ == "__main__":
    import sys

    _write_combinations(sys.argv[1])
