import typing

from metaflow import (
    FlowSpec,  # pyright: ignore [reportPrivateImportUsage]
    Parameter,  # pyright: ignore [reportPrivateImportUsage]
    conda_base,  # pyright: ignore [reportAttributeAccessIssue]
    current,  # pyright: ignore [reportPrivateImportUsage]
    step,  # pyright: ignore [reportPrivateImportUsage]
)

__OUTPUT_ARTIFACTS__ = ("ephemeral", "games_ds")

__DATASETS__ = (
    "adversaries_ds",
    "spirits_ds",
    "matchups_ds",
    "combinations_ds",
)


@conda_base(python=">=3.12,<3.13", packages={"polars": "==1.1.0"})
class SugrGamesFlow(FlowSpec):
    param_input = Parameter("input", required=True, type=str)
    param_keep = Parameter("keep", default=False)
    param_player_limit = Parameter("player-limit", default=6)
    # TODO: Set this to the real defaults
    param_subset = Parameter("subset", default=True)

    @step
    def start(self) -> None:
        import os
        from pathlib import Path

        import polars as pl
        from utilities.hive_dataset import HiveDataset
        from utilities.working_dir import WorkingDirectory

        input_dir = Path(typing.cast(str, self.param_input))
        self.expansions_tsv = input_dir / "expansions.tsv"
        self.spirits_tsv = input_dir / "spirits.tsv"
        self.adversaries_tsv = input_dir / "adversaries.tsv"
        self.escalations_tsv = input_dir / "escalations.tsv"

        temp = WorkingDirectory.for_metaflow_run(
            "sugr-games",
            typing.cast(int, current.run_id),
        )
        self.games_ds = HiveDataset(
            temp.push_segment("results").path,
            "games",
            Expansion=pl.UInt8,  # type: ignore [argumentType]
            Players=pl.UInt8,  # type: ignore [argumentType]
            Difficulty=pl.UInt8,  # type: ignore [argumentType]
            Complexity=pl.UInt8,  # type: ignore [argumentType]
        )

        self.ephemeral = temp.push_segment("ephemeral")
        os.environ["POLARS_TEMP_DIR"] = str(
            self.ephemeral.push_segment(
                "polars",
            ),
        )

        self.next(self.fanout_expansions)

    @step
    def fanout_expansions(self) -> None:
        import polars as pl
        from utilities.hive_dataset import HiveDataset

        exp = pl.scan_csv(self.expansions_tsv, separator="\t")
        if self.param_subset:
            exp = exp.filter(pl.col("Value").is_in([1, 2, 13, 14, 31]))
        self.expansions = exp.select("Value", "Players").collect(streaming=True).rows()

        self.adversaries_ds = HiveDataset(
            self.ephemeral.path,
            "adversaries",
            Expansion=pl.UInt8,  # type: ignore [argumentType]
        )
        self.spirits_ds = HiveDataset(
            self.ephemeral.path,
            "spirits",
            Expansion=pl.UInt8,  # type: ignore [argumentType]
        )
        self.matchups_ds = HiveDataset(
            self.ephemeral.path,
            "matchups",
            Expansion=pl.UInt8,  # type: ignore [argumentType]
            Matchup=pl.String,  # type: ignore [argumentType]
        )
        self.combinations_ds = HiveDataset(
            self.ephemeral.path,
            "combinations",
            Expansion=pl.UInt8,  # type: ignore [argumentType]
            Players=pl.UInt8,  # type: ignore [argumentType]
            Matchup=pl.String,  # type: ignore [argumentType]
        )
        self.next(self.filter_by_expansion, foreach="expansions")

    @step
    def filter_by_expansion(self) -> None:
        import polars as pl

        from transformations.sugr.adversaries import adversaries_by_expansions
        from transformations.sugr.spirits import spirits_by_expansions

        (self.expansion, self.max_players) = typing.cast(
            tuple[int, int],
            self.input,
        )
        self.max_players = min(
            self.max_players,
            typing.cast(int, self.param_player_limit),
        )

        (adversaries, self.matchups) = adversaries_by_expansions(
            self.expansion,
            pl.scan_csv(self.adversaries_tsv, separator="\t"),
            pl.scan_csv(self.escalations_tsv, separator="\t"),
        )
        self.adversaries_ds.write(adversaries, Expansion=self.expansion)

        self.spirits_ds.write(
            spirits_by_expansions(
                self.expansion,
                pl.scan_csv(self.spirits_tsv, separator="\t"),
            ),
            Expansion=self.expansion,
        )

        self.next(self.fanout_matchups)

    @step
    def fanout_matchups(self) -> None:
        self.next(self.calculate_matchups, foreach="matchups")

    @step
    def calculate_matchups(self) -> None:
        from transformations.sugr.spirits import calculate_matchups

        self.matchup = typing.cast(str, self.input)

        self.matchups_ds.write(
            calculate_matchups(
                self.matchup,
                self.spirits_ds.read(Expansion=self.expansion),
            ),
            Expansion=self.expansion,
            Matchup=self.matchup,
        )

        self.next(self.generate_combinations)

    @step
    def generate_combinations(self) -> None:
        from transformations.sugr.spirits import generate_combinations

        for pc in range(self.max_players):
            if pc == 0:
                self.combinations_ds.write(
                    generate_combinations(
                        self.matchups_ds.read(
                            Expansion=self.expansion,
                            Matchup=self.matchup,
                        ),
                    ),
                    Expansion=self.expansion,
                    Players=1,
                    Matchup=self.matchup,
                )
                continue

            self.combinations_ds.write(
                generate_combinations(
                    self.matchups_ds.read(
                        Expansion=self.expansion,
                        Matchup=self.matchup,
                    ),
                    self.combinations_ds.read(
                        Expansion=self.expansion,
                        Matchup=self.matchup,
                        Players=pc,
                    ),
                ),
                Expansion=self.expansion,
                Players=pc + 1,
                Matchup=self.matchup,
            )

        self.next(self.collect_matchups)

    @step
    def collect_matchups(self, inputs: typing.Any) -> None:
        self.merge_artifacts(
            inputs,
            include=[*__OUTPUT_ARTIFACTS__, *__DATASETS__],
        )
        self.next(self.collect_expansions)

    @step
    def collect_expansions(self, inputs: typing.Any) -> None:
        self.merge_artifacts(
            inputs,
            include=[*__OUTPUT_ARTIFACTS__, *__DATASETS__],
        )
        self.next(self.fanout_buckets)

    @step
    def fanout_buckets(self) -> None:
        from transformations.sugr.games import define_buckets

        self.buckets = define_buckets(
            self.adversaries_ds.read(),
            self.combinations_ds.read(),
        )

        self.next(self.bucket_games, foreach="buckets")

    @step
    def bucket_games(self) -> None:
        from transformations.sugr.games import filter_by_bucket

        (
            expansion,
            players,
            difficulty,
            difficulty_min,
            difficulty_max,
            complexity,
            complexity_min,
            complexity_max,
        ) = typing.cast(
            tuple[int, int, int, float, float, int, float, float],
            self.input,
        )

        self.games_ds.write(
            filter_by_bucket(
                (difficulty_min, difficulty_max),
                (complexity_min, complexity_max),
                self.adversaries_ds.read(Expansion=expansion),
                self.combinations_ds.read(
                    Expansion=expansion,
                    Players=players,
                ),
            ),
            Expansion=expansion,
            Players=players,
            Difficulty=difficulty,
            Complexity=complexity,
        )

        self.next(self.collect_buckets)

    @step
    def collect_buckets(self, inputs: typing.Any) -> None:
        self.merge_artifacts(inputs, include=[*__OUTPUT_ARTIFACTS__])
        self.next(self.end)

    @step
    def end(self) -> None:
        if not self.param_keep:
            self.ephemeral.cleanup()


if __name__ == "__main__":
    SugrGamesFlow()
