from typing import Literal
import polars as pl


def generate_board_combinations(
    total_boards: Literal[4] | Literal[6], players: int
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
    """Takes a given set of layouts/weights and turns them into a randomized, selectable list for the player count."""
    import random

    player_layouts = (
        layouts.clone()
        .filter(pl.col("Players").eq(pl.lit(players)))
        .collect(streaming=True)
    )

    return pl.DataFrame(
        random.choices(
            player_layouts.select("Layout", "Standard").rows(),
            weights=[r[0] for r in player_layouts.select("Weight").iter_rows()],
            k=100,
        ),
        {"Layout": pl.Utf8, "Standard": pl.Boolean},
    ).lazy()


def generate_loose_islands(layouts: pl.LazyFrame, boards: pl.LazyFrame) -> pl.LazyFrame:
    """Generates all possible islands combing layouts + boards. This frame is sorted with the standard layout first."""
    return (
        layouts.clone()
        .join(boards, how="cross")
        .sort(pl.col("Standard"), descending=True)
        .select(["Layout", "Boards", "Standard"])
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

    raise Exception("Only 1-3 players are supported on the fixed island board.")
