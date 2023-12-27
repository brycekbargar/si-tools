import polars as pl
import time
from rich.console import Console
from pathlib import Path
import gc
from typing import cast, Any
import fastparquet as fp


class State:
    def __init__(self, root: str):
        self.console = Console()
        self.run = time.time_ns()
        self.console.log(f"Run: {self.run}")

        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.output = self.root / "results" / str(self.run)
        self.output.mkdir(parents=True, exist_ok=True)
        self.temp = self.root / "temp" / str(self.run)
        self.temp.mkdir(parents=True, exist_ok=True)

    def reset(self, row: dict[str, Any]):
        self.expansions = cast(int, row["Value"])
        self.console.print()
        self.console.rule(f"{self.run} / {self.expansions}", align="left")
        self.console.log(cast(str, row["Expansions"]), justify="center")
        self.console.print()

    def log_rule(self, message: str):
        self.console.rule(message, align="right")

    def log_indent(self, indent: int, message: str):
        self.console.log("\t" * indent + " " + message)

    def check(self, indent: int, rtype: str, loc: Path):
        records = fp.ParquetFile(loc, verify=True)
        self.log_indent(indent, f"{records.count()} {rtype}: {records.dtypes}")

    def temp_spirits(self):
        return self.temp / f"{self.expansions:02}_spirits.parquet"

    def temp_adversaries(self):
        return self.temp / f"{self.expansions:02}_adversaries.parquet"

    def set_matchup(self, matchup: str):
        self.matchup = matchup
        self.log_indent(0, self.matchup)

    def temp_matchups(self):
        return self.temp / f"{self.expansions:02}_{self.matchup}_matchups.parquet"

    def temp_combinations(self, players: int):
        return self.temp / f"{self.expansions:02}_{self.matchup}_{players:02}.parquet"


def main():
    expansions_tsv = pl.read_csv(
        "./data/expansions.tsv",
        separator="\t",
        dtypes={"Value": pl.Int8, "Players": pl.Int8},
    )
    spirits_tsv = pl.scan_csv(
        "./data/spirits.tsv",
        separator="\t",
        dtypes={"Expansion": pl.Int8},
    )
    adversaries_tsv = pl.scan_csv(
        "./data/adversaries.tsv",
        separator="\t",
        dtypes={
            "Difficulty": pl.Int8,
            "Complexity": pl.Int8,
            "Level": pl.Int8,
            "Expansion": pl.Int8,
        },
    )
    escalations_tsv = pl.scan_csv(
        "./data/escalations.tsv",
        separator="\t",
        dtypes={
            "Difficulty": pl.Int8,
            "Complexity": pl.Int8,
            "Expansion": pl.Int8,
        },
    )

    state = State("./data")
    for exp_row in expansions_tsv.rows(named=True):
        if cast(int, exp_row["Value"]) not in [1, 2, 11]:
            # continue
            pass

        state.reset(exp_row)

        state.log_rule("Spirits / Expansions")
        spirits = (
            spirits_tsv.clone()
            .filter(pl.col("Expansions").or_(state.expansions).eq(state.expansions))
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
                .alias("Aspect")
            )
            .drop("count")
        )
        spirits = (
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
        spirits.sink_parquet(state.temp_spirits(), maintain_order=False)

        state.check(0, "Spirits", state.temp_spirits())
        del spirits
        gc.collect()

        adversaries = (
            adversaries_tsv.clone()
            .filter(pl.col("Expansion").or_(state.expansions).eq(state.expansions))
            .drop("Expansion")
            .rename({"Name": "Adversary"})
        )
        if (
            adversaries.clone().select(pl.col("Adversary").n_unique()).collect().item()
            <= 3
        ):
            adversaries = pl.concat([adversaries, escalations_tsv], how="diagonal")

        adversaries.sink_parquet(state.temp_adversaries(), maintain_order=False)

        matchups = [
            cast(str, m[0])
            for m in adversaries.clone()
            .unique(subset=["Matchup"])
            .select(pl.col("Matchup"))
            .collect()
            .rows()
        ]
        state.log_indent(0, ", ".join(matchups))

        state.check(0, "Adversaries", state.temp_adversaries())
        del adversaries
        gc.collect()

        state.log_rule("Matchups / Combinations")

        for matchup in matchups:
            state.set_matchup(matchup)

            spirit_matchups = (
                pl.scan_parquet(state.temp_spirits())
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
                .with_columns(
                    pl.col("Aspect").count().over("Name").name.suffix(" Count")
                )
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
                        ]
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
                        ]
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
                                ]
                            )
                        )
                        .alias("Spirit"),
                    )
                )
            spirit_matchups = spirit_matchups.select(
                pl.col("Difficulty"),
                pl.col("Complexity"),
                pl.col("Spirit"),
            )
            # sink_parquet doesn't support the list/string munging for aspects
            spirit_matchups.collect(streaming=True).write_parquet(state.temp_matchups())

            state.check(1, f"Matchups for {matchup}", state.temp_matchups())
            del spirit_matchups
            gc.collect()

            (
                pl.scan_parquet(state.temp_matchups())
                .with_columns(
                    [
                        pl.lit(matchup).alias("Matchup"),
                        pl.col("Complexity").alias("NComplexity"),
                        pl.col("Spirit").hash().alias("Hash"),
                    ]
                )
                .rename({"Spirit": "Spirit_0"})
                .select(
                    pl.col("Matchup"),
                    pl.col("Difficulty"),
                    pl.col("NComplexity"),
                    pl.col("Complexity"),
                    pl.col("Spirit_0"),
                    pl.col("Hash"),
                )
                .sink_parquet(state.temp_combinations(1), maintain_order=False)
            )

            state.check(2, "Combinations for 1 player", state.temp_combinations(1))
            gc.collect()

            for pi in range(1, cast(int, exp_row["Players"])):
                sp_col = f"Spirit_{pi}"
                players = pi + 1

                def _unique_spirits() -> pl.Expr:
                    spirit_n = pl.col(sp_col)
                    expr = pl.Expr.not_(spirit_n.eq(pl.col("Spirit_0")))
                    for i in range(pi - 1, 0, -1):
                        expr = expr.and_(
                            pl.Expr.not_(spirit_n.eq(pl.col(f"Spirit_{i}")))
                        )
                    return expr

                (
                    pl.scan_parquet(state.temp_matchups())
                    .join(
                        pl.scan_parquet(state.temp_combinations(players - 1)),
                        how="cross",
                    )
                    .with_columns(
                        [
                            pl.col("Difficulty").add(pl.col("Difficulty_right")),
                            pl.col("Complexity").add(pl.col("Complexity_right")),
                            pl.col("Hash").add(pl.col("Spirit").hash()),
                            pl.col("Complexity")
                            .truediv(players)
                            .round()
                            .cast(pl.Int8)
                            .alias("NComplexity"),
                        ]
                    )
                    .rename({"Spirit": sp_col})
                    .drop("Difficulty_right", "Complexity_right")
                    .filter(_unique_spirits())
                    .sort("NComplexity", descending=True)
                    .unique(subset="Hash", keep="first")
                ).sink_parquet(state.temp_combinations(players), maintain_order=False)

                state.check(
                    2,
                    f"Combinations for {players} players",
                    state.temp_combinations(players),
                )
                gc.collect()


if __name__ == "__main__":
    main()
