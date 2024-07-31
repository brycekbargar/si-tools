"""Provides operations on LazyFrames to generate island setups."""

from typing import Literal

import polars as pl


def generate_board_combinations(
    total_boards: Literal[4, 6, 8],
    players: int,
) -> pl.LazyFrame:
    """Generates all valid combinations of boards for the number of players."""
    from functools import reduce
    from itertools import combinations

    boards = {
        0b00000001: "A",
        0b00000010: "B",
        0b00000100: "C",
        0b00001000: "D",
    }
    if total_boards >= 6:
        boards |= {
            0b00010000: "E",
            0b00100000: "F",
        }
    if total_boards == 8:
        boards |= {
            0b01000000: "G",
            0b10000000: "H",
        }

    combos = [
        (reduce(lambda c, b: c | b, list(p), 0b0), list(p))
        for p in combinations(list(boards.keys()), players)
    ]
    if players < 4:
        combos = [
            c
            for c in combos
            if (c[0] & 0b00010010 != 0b00010010)
            and (c[0] & 0b00101000 != 0b00101000)
            and (c[0] & 0b01000100 != 0b01000100)
            and (c[0] & 0b10000001 != 0b10000001)
        ]

    return pl.LazyFrame(
        [[boards[b] for b in c[1]] for c in combos],
        {f"Board_{p}": pl.Utf8 for p in range(players)},
    )


def explode_layouts(layouts: pl.LazyFrame, players: int) -> pl.LazyFrame:
    """Returns a randomized list of layouts based on input weights."""
    import random

    player_layouts = (
        layouts.clone()
        .filter(pl.col("Players").eq(pl.lit(players)))
        .collect(streaming=True)
    )

    def _generate() -> pl.DataFrame:
        return pl.DataFrame(
            random.choices(  # noqa: S311
                player_layouts.select("Players", "Layout").rows(),
                weights=[r[0] for r in player_layouts.select("Weight").iter_rows()],
                k=20,
            ),
            ["Players", "Layout"],
            orient="row",
        )

    expected = player_layouts.select(
        pl.col("Layout"),
        pl.col("Weight").truediv(pl.sum("Weight")).alias("Expected"),
    ).to_dict(as_series=False)

    tries = 0
    while tries < 50:
        possible = _generate()
        actual = (
            possible.group_by("Layout")
            .agg(pl.len().alias("Actual").truediv(pl.lit(20)))
            .to_dict(as_series=False)
        )

        within_bounds = len(expected["Layout"]) == len(actual["Layout"])
        for i in range(len(expected["Layout"])):
            if not within_bounds:
                break

            layout = expected["Layout"][i]
            expected_percent = expected["Expected"][i]
            actual_percent = actual["Actual"][actual["Layout"].index(layout)]

            if abs(expected_percent - actual_percent) >= 0.03:
                within_bounds = False

        if within_bounds:
            return possible.with_columns(pl.lit(players).alias("Players")).lazy()

        tries += 1

    unlucky = f"Wasn't able to generate layout distribution after {tries} tries"
    raise GeneratorExit(unlucky)


def generate_loose_islands(layouts: pl.LazyFrame, boards: pl.LazyFrame) -> pl.LazyFrame:
    """Generates all possible islands combing layouts + boards."""
    return layouts.clone().join(boards.clone(), on="Players")


def generate_fixed_islands(players: int) -> pl.LazyFrame:
    """Hackily generates layouts + boards for Horizons only games."""
    match players:
        case 1:
            return pl.LazyFrame(
                [
                    ["Three-Player Side", "F"],
                    ["Three-Player Side", "G"],
                    ["Three-Player Side", "H"],
                ],
                ["Layout", "Board_0"],
            )
        case 2:
            return pl.LazyFrame(
                [
                    ["Three-Player Side", "F", "G"],
                    ["Three-Player Side", "F", "H"],
                    ["Three-Player Side", "G", "F"],
                    ["Three-Player Side", "G", "H"],
                    ["Three-Player Side", "H", "F"],
                    ["Three-Player Side", "H", "G"],
                    ["Two-Player Side", "G", "H"],
                    ["Two-Player Side", "H", "G"],
                ],
                ["Layout", "Board_0", "Board_1"],
            )
        case 3:
            return pl.LazyFrame(
                [
                    ["Three-Player Side", "F", "G", "H"],
                    ["Three-Player Side", "F", "H", "G"],
                    ["Three-Player Side", "G", "F", "H"],
                    ["Three-Player Side", "G", "H", "F"],
                    ["Three-Player Side", "H", "F", "G"],
                    ["Three-Player Side", "H", "G", "F"],
                ],
                ["Layout", "Board_0", "Board_1", "Board_2"],
            )
        case _:
            msg = "Only 1-3 players are supported on the fixed island board."
            raise IndexError(msg)
