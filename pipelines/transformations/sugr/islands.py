"""Provides operations on LazyFrames to generate island setups."""

from typing import Literal

import polars as pl


def generate_board_combinations(
    total_boards: Literal[4, 6, 8],
    players: int,
) -> pl.LazyFrame:
    """Generates all valid combinations of boards for the number of players."""
    from functools import reduce
    from itertools import permutations

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
        for p in permutations(list(boards.keys()), players)
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

    return (
        pl.LazyFrame(
            [[boards[b] for b in c[1]] for c in combos],
            {f"Board_{p}": pl.Utf8 for p in range(players)},
        )
        .with_columns(pl.lit(players).cast(pl.UInt8).alias("Players"))
        .with_columns(pl.lit(f"{total_boards}B").alias("Type"))
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
                player_layouts.select("Layout").rows(),
                weights=[r[0] for r in player_layouts.select("Weight").iter_rows()],
                k=100,
            ),
            {"Layout": pl.Utf8},
        ).with_columns(pl.lit(players).cast(pl.UInt8).alias("Players"))

    expected = player_layouts.select(
        pl.col("Layout"),
        pl.col("Weight").truediv(pl.sum("Weight")).alias("Expected"),
    ).to_dict(as_series=False)

    tries = 0
    while tries < 50:
        possible = _generate()
        actual = (
            possible.group_by("Layout")
            .agg(pl.len().alias("Actual").truediv(pl.lit(100)))
            .to_dict(as_series=False)
        )

        within_bounds = True
        for i in range(len(expected["Layout"])):
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
    """Generates all possible islands combing layouts + boards."""
    #  TODO: Remove this casting hack
    return layouts.clone().join(boards.cast({"Players": pl.UInt8}), on="Players")


def generate_fixed_islands(players: int) -> pl.LazyFrame:
    """Hackily generates layouts + boards for Horizons only games."""
    frame: pl.LazyFrame
    match players:
        case 1:
            frame = pl.LazyFrame(
                [
                    ["FB", 1, "Three-Player Side", "F"],
                    ["FB", 1, "Three-Player Side", "G"],
                    ["FB", 1, "Three-Player Side", "H"],
                ],
                ["Type", "Players", "Layout", "Board_0"],
            )
        case 2:
            return pl.LazyFrame(
                [
                    ["FB", 2, "Three-Player Side", "F", "G"],
                    ["FB", 2, "Three-Player Side", "F", "H"],
                    ["FB", 2, "Three-Player Side", "G", "F"],
                    ["FB", 2, "Three-Player Side", "G", "H"],
                    ["FB", 2, "Three-Player Side", "H", "F"],
                    ["FB", 2, "Three-Player Side", "H", "G"],
                    ["FB", 2, "Two-Player Side", "G", "H"],
                    ["FB", 2, "Two-Player Side", "H", "G"],
                ],
                ["Type", "Players", "Layout", "Board_0", "Board_1"],
            )
        case 3:
            return pl.LazyFrame(
                [
                    ["FB", 3, "Three-Player Side", "F", "G", "H"],
                    ["FB", 3, "Three-Player Side", "F", "H", "G"],
                    ["FB", 3, "Three-Player Side", "G", "F", "H"],
                    ["FB", 3, "Three-Player Side", "G", "H", "F"],
                    ["FB", 3, "Three-Player Side", "H", "F", "G"],
                    ["FB", 3, "Three-Player Side", "H", "G", "F"],
                ],
                ["Type", "Players", "Layout", "Board_0", "Board_1", "Board2"],
            )
        case _:
            msg = "Only 1-3 players are supported on the fixed island board."
            raise IndexError(msg)

    return frame.cast({"Players": pl.UInt8})
