# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
# %conda install polars --yes

# %%
import polars as pl

if hasattr(__builtins__, "__IPYTHON__"):
    from rich import inspect, print

    inspect("ruff/isort strip out rich")
    print("unless it is used")


# %%
def spirits_by_expansions(expansions: int, spirits: pl.LazyFrame) -> pl.LazyFrame:
    """Filter, and clean Spirit data."""
    import polars as pl

    # Filter to just the given expansions.
    # Expansions is a bitfield, this is assuming that a superset of the expansions for a spirit are required.
    spirits = spirits.filter(pl.col("Expansions").or_(expansions).eq(expansions)).drop(
        "Expansions"
    )

    # Cleanup the Aspect column.
    # We only care to label spirits as the "Base" version when aspects are available.
    spirits = (
        spirits.join(spirits.group_by("Name").count(), on="Name", how="left")
        .with_columns(
            pl.when(pl.col("Aspect").is_not_null())
            .then(pl.col("Aspect"))
            .when(pl.col("count").gt(1))
            .then(pl.lit("Base"))
            .otherwise(None)
            .alias("Aspect")
        )
        .drop("count")
    )

    # Convert Complexity from text to numeric.
    spirits = (
        spirits.join(
            pl.LazyFrame(
                {
                    "Complexity": ["Low", "Moderate", "High", "Very High"],
                    "Value": [0, 1, 2, 4],
                }
            ),
            on="Complexity",
            how="left",
        )
        .with_columns(pl.col("Value").alias("Complexity"))
        .drop("Value")
    )

    return spirits


# %%
if hasattr(__builtins__, "__IPYTHON__"):
    all_spirits = spirits_by_expansions(
        63, pl.scan_csv("../data/spirits.tsv", separator="\t")
    )


# %%
def calculate_matchups(matchup: str, spirits: pl.LazyFrame) -> pl.LazyFrame:
    """Calculate the difficulty modifiers and best spirits for the matchup."""
    import polars as pl

    # Convert matchhup from text to numeric difficulty.
    matchups = (
        spirits.filter(pl.Expr.not_(pl.col(matchup).eq(pl.lit("Unplayable"))))
        .join(
            pl.LazyFrame(
                {
                    matchup: [
                        "Top",
                        "Counters",
                        "Mid+",
                        "Neutral",
                        "Mid-",
                        "Bottom",
                        "Unfavored",
                    ],
                    "Difficulty": [-1, -1, 0, 0, 1, 2, 2],
                }
            ),
            on=matchup,
            how="left",
        )
        .select(
            pl.col("Name"),
            pl.col("Difficulty"),
            pl.col("Aspect"),
            pl.col("Complexity"),
        )
    )

    # Find the best aspects.
    matchups = (
        matchups.group_by(["Name", "Difficulty"])
        .agg([pl.col("Aspect"), pl.max("Complexity")])
        .filter(pl.col("Difficulty").eq(pl.min("Difficulty").over("Name")))
        .with_columns(
            pl.when(pl.col("Aspect").list.drop_nulls().list.len().eq(0))
            .then(pl.col("Name"))
            .otherwise(
                pl.concat_str(
                    [
                        pl.col("Name"),
                        pl.lit(" ("),
                        pl.col("Aspect").list.join(", "),
                        pl.lit(")"),
                    ]
                )
            )
            .alias("Spirit")
        )
    )

    # Reorder and clean up columns.
    matchups = matchups.with_columns(pl.lit(matchup).alias("Matchup")).select(
        pl.col("Matchup"), pl.col("Difficulty"), pl.col("Complexity"), pl.col("Spirit")
    )

    return matchups


# %%
if hasattr(__builtins__, "__IPYTHON__"):
    matchups = calculate_matchups("Sweden", all_spirits)


# %%
def generate_combinations(count: int, matchups: pl.DataFrame) -> pl.DataFrame:
    """Generate all possible combinations of spirits."""
    import polars as pl

    def _unique_spirits(sc: int) -> pl.Expr:
        """Check the latest cross-joined spirit column against all the previous ones."""
        spirit_n = pl.col(f"Spirit_{sc}")
        expr = pl.lit(1).eq(pl.lit(1))
        for i in range(sc - 1, -1, -1):
            expr = expr.and_(pl.Expr.not_(spirit_n.eq(pl.col(f"Spirit_{i}"))))
        return expr

    combos = matchups.clone().rename({"Spirit": "Spirit_0"})
    matchups = matchups.drop("Matchup")
    for i in range(1, count):
        combos = (
            combos.join(matchups, how="cross")
            .with_columns(
                [
                    pl.col("Difficulty").add(pl.col("Difficulty_right")),
                    pl.col("Complexity").add(pl.col("Complexity_right")),
                ]
            )
            .drop("Difficulty_right", "Complexity_right")
            .rename({"Spirit": f"Spirit_{i}"})
            .filter(_unique_spirits(i))
        )

    return combos


# %%
if hasattr(__builtins__, "__IPYTHON__"):
    print(generate_combinations(2, matchups).collect())

# %%
