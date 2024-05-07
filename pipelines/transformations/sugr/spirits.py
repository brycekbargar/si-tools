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

# ruff: noqa

# %%
# %mamba install polars --yes --quiet

# %%

import polars as pl


# %%
def spirits_by_expansions(expansions: int, spirits: pl.LazyFrame) -> pl.LazyFrame:
    """Filter, and clean Spirit data."""
    spirits = (
        spirits.clone()
        .filter(pl.col("Expansions").or_(expansions).eq(expansions))
        .drop("Expansions")
    )

    spirits = (
        spirits.join(spirits.group_by("Name").count(), on="Name", how="left")
        .with_columns(
            pl.when(pl.col("Aspect").is_not_null())
            .then(pl.col("Aspect"))
            .when(pl.col("count").gt(1))
            .then(pl.lit("Base"))
            .otherwise(None)
            .alias("Aspect"),
        )
        .drop("count")
    )

    return (
        spirits.join(
            pl.LazyFrame(
                {
                    "Complexity": ["Low", "Moderate", "High", "Very High"],
                    "Value": [0, 1, 2, 4],
                },
                schema={
                    "Complexity": None,
                    "Value": pl.Int8,
                },
            ),
            on="Complexity",
            how="left",
        )
        .with_columns(pl.col("Value").alias("Complexity"))
        .drop("Value")
    )


# %%
if hasattr(__builtins__, "__IPYTHON__"):
    all_spirits = spirits_by_expansions(
        63,
        pl.scan_csv("../data/spirits.tsv", separator="\t"),
    )


# %%
def calculate_matchups(matchup: str, spirits: pl.LazyFrame) -> pl.LazyFrame:
    """Calculate the difficulty modifiers and best/worst spirits for the matchup."""
    spirit_matchups = (
        spirits.clone()
        .join(
            pl.LazyFrame(
                {
                    matchup: [
                        "Counters",
                        "Neutral",
                        "Unfavored",
                        "Unplayable",
                        "Top",
                        "Mid+",
                        "Mid-",
                        "Bottom",
                    ],
                    "Difficulty": [-1, 0, 2, 99, -2, -1, 0, 2],
                },
                schema={
                    matchup: None,
                    "Difficulty": pl.Int8,
                },
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
        .with_columns(pl.col("Aspect").count().over("Name").name.suffix(" Count"))
    )

    if matchup == "Tier":
        spirit_matchups = (
            spirit_matchups.clone()
            .group_by("Name")
            .agg(
                [
                    pl.max("Aspect Count"),
                    pl.min("Difficulty"),
                    pl.max("Complexity"),
                ],
            )
            .with_columns(
                pl.when(pl.col("Aspect Count").eq(0))
                .then(pl.col("Name"))
                .otherwise(pl.concat_str([pl.col("Name"), pl.lit(" (Any)")]))
                .alias("Spirit"),
            )
        )

    else:
        spirit_matchups = (
            spirit_matchups.clone()
            .filter(pl.Expr.not_(pl.col("Difficulty").eq(99)))
            .group_by(["Name", "Difficulty"])
            .agg(
                [
                    pl.col("Aspect"),
                    pl.max("Aspect Count"),
                    pl.max("Complexity"),
                ],
            )
            .filter(pl.col("Difficulty").eq(pl.min("Difficulty").over("Name")))
            .with_columns(
                pl.when(pl.col("Aspect Count").eq(0))
                .then(pl.col("Name"))
                .otherwise(
                    pl.concat_str(
                        [
                            pl.col("Name"),
                            pl.lit(" ("),
                            pl.col("Aspect").list.join(", "),
                            pl.lit(")"),
                        ],
                    ),
                )
                .alias("Spirit"),
            )
        )

    spirit_matchups = spirit_matchups.select(
        pl.col("Difficulty"),
        pl.col("Complexity"),
        pl.col("Spirit"),
    )

    # list/string munging isn't supported by sink_parquet as of 0.20.2
    return spirit_matchups.collect(streaming=True).lazy()


# %%
if hasattr(__builtins__, "__IPYTHON__"):
    matchups = calculate_matchups("Sweden", all_spirits)


# %%
def generate_combinations(
    matchup: str,
    players: int,
    matchups: pl.LazyFrame,
    previous_combos: pl.LazyFrame | None = None,
) -> pl.LazyFrame:
    """Generate all possible combinations of spirits."""
    if players == 1:
        return (
            matchups.clone()
            .with_columns(
                [
                    pl.lit(matchup).alias("Matchup"),
                    pl.col("Difficulty").cast(pl.Float32).alias("NDifficulty"),
                    pl.col("Complexity").cast(pl.Float32).alias("NComplexity"),
                    pl.col("Spirit").hash().alias("Hash"),
                ],
            )
            .rename({"Spirit": "Spirit_0"})
            .select(
                pl.col("Matchup"),
                pl.col("NDifficulty"),
                pl.col("Difficulty"),
                pl.col("NComplexity"),
                pl.col("Complexity"),
                pl.col("Spirit_0"),
                pl.col("Hash"),
            )
        )

    if previous_combos is None:
        msg = "Requires previously generated combos for 2+ players"
        raise Exception(msg)

    sp_col = f"Spirit_{(players-1)}"

    def _unique_spirits() -> pl.Expr:
        spirit_n = pl.col(sp_col)
        expr = pl.Expr.not_(spirit_n.eq(pl.col("Spirit_0")))
        for i in range(players - 2, 0, -1):
            expr = expr.and_(pl.Expr.not_(spirit_n.eq(pl.col(f"Spirit_{i}"))))
        return expr

    return (
        matchups.clone()
        .join(
            previous_combos.clone(),
            how="cross",
        )
        .with_columns(
            [
                pl.col("Difficulty").add(pl.col("Difficulty_right")),
                pl.col("Complexity").add(pl.col("Complexity_right")),
                pl.col("Hash").add(pl.col("Spirit").hash()),
            ],
        )
        .with_columns(
            pl.col("Difficulty").truediv(players).cast(pl.Float32).alias("NDifficulty"),
            pl.col("Complexity").truediv(players).cast(pl.Float32).alias("NComplexity"),
        )
        .rename({"Spirit": sp_col})
        .drop("Difficulty_right", "Complexity_right")
        .filter(_unique_spirits())
        .sort("NComplexity", descending=True)
        .unique(subset="Hash", keep="first")
    )


# %%
if hasattr(__builtins__, "__IPYTHON__"):
    combos = generate_combinations(
        "Sweden",
        2,
        matchups,
        generate_combinations("Sweden", 1, matchups),
    )

# %%
