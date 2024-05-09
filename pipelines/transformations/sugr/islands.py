"""Provides operations on LazyFrames to generate island setups."""

from typing import Literal

import polars as pl


def generate_board_combinations(
    total_boards: Literal[4, 6],
    players: int,
) -> pl.LazyFrame:
    """Generates all valid combinations of boards for the number of players."""
    from functools import reduce
    from itertools import permutations

    boards = {
        0b000001: "A",
        0b000010: "B",
        0b000100: "C",
        0b001000: "D",
    }
    if total_boards == 6:
        boards |= {
            0b010000: "E",
            0b100000: "F",
        }

    combos = [
        (reduce(lambda c, b: c | b, list(p), 0b0), list(p))
        for p in permutations(list(boards.keys()), players)
    ]
    if players < 4:
        combos = [
            c
            for c in combos
            if (c[0] & 0b010010 != 0b010010) and (c[0] & 0b101000 != 0b101000)
        ]

    return pl.LazyFrame(
        [[boards[b] for b in c[1]] for c in combos],
        {f"Board_{p}": pl.Utf8 for p in range(players)},
    ).with_columns(pl.lit(players).cast(pl.UInt8).alias("Players"))


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
                player_layouts.select("Layout", "Standard").rows(),
                weights=[r[0] for r in player_layouts.select("Weight").iter_rows()],
                k=100,
            ),
            {"Layout": pl.Utf8, "Standard": pl.Boolean},
        )

    expected = player_layouts.select(
        pl.col("Layout"),
        pl.col("Weight").truediv(pl.sum("Weight")).alias("Expected"),
    ).to_dict(as_series=False)

    tries = 0
    while tries < 10:
        possible = _generate()
        actual = (
            possible.group_by("Layout")
            .agg(pl.len().alias("Actual").truediv(pl.lit(100)))
            .to_dict(as_series=False)
        )

        within_bounds = True
        for i in range(len(expected)):
            if not within_bounds:
                continue

            layout = expected["Layout"][i]
            expected_percent = expected["Expected"][i]
            actual_percent = actual["Actual"][actual["Layout"].index(layout)]

            if abs(expected_percent - actual_percent) >= 0.03:
                within_bounds = False

        if within_bounds:
            return possible.lazy()

        tries += 1

    unlucky = f"Wasn't able to generate layout distribution after {tries} tries"
    raise GeneratorExit(unlucky)


def generate_loose_islands(layouts: pl.LazyFrame, boards: pl.LazyFrame) -> pl.LazyFrame:
    """Generates all possible islands combing layouts + boards.

    This frame is sorted with the standard layout first.
    """
    return (
        layouts.clone()
        .join(boards, how="cross")
        .sort(pl.col("Standard"), descending=True)
    )


def generate_fixed_islands(players: int) -> pl.LazyFrame:
    """Hackily generates layouts + boards for Horizons only games."""
    match players:
        case 1:
            return pl.LazyFrame(
                [
                    ["Three-Player Side", True, "F"],
                    ["Three-Player Side", True, "G"],
                    ["Three-Player Side", True, "H"],
                ],
                ["Layout", "Standard", "Board_0"],
            )
        case 2:
            return pl.LazyFrame(
                [
                    ["Three-Player Side", False, "F", "G"],
                    ["Three-Player Side", False, "F", "H"],
                    ["Three-Player Side", False, "G", "F"],
                    ["Three-Player Side", False, "G", "H"],
                    ["Three-Player Side", False, "H", "F"],
                    ["Three-Player Side", False, "H", "G"],
                    ["Two-Player Side", True, "G", "H"],
                    ["Two-Player Side", True, "H", "G"],
                ],
                ["Layout", "Standard", "Board_0", "Board_1"],
            )
        case 3:
            return pl.LazyFrame(
                [
                    ["Three-Player Side", True, "F", "G", "H"],
                    ["Three-Player Side", True, "F", "H", "G"],
                    ["Three-Player Side", True, "G", "F", "H"],
                    ["Three-Player Side", True, "G", "H", "F"],
                    ["Three-Player Side", True, "H", "F", "G"],
                    ["Three-Player Side", True, "H", "G", "F"],
                ],
                ["Layout", "Standard", "Board_0", "Board_1", "Board2"],
            )

    msg = "Only 1-3 players are supported on the fixed island board."
    raise IndexError(msg)
