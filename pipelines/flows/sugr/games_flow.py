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
    "input_expansions_ds",
    "input_adversaries_ds",
    "input_escalations_ds",
    "input_spirits_ds",
    "adversaries_ds",
    "spirits_ds",
    "matchups_ds",
    "combinations_ds",
)


@conda_base(python=">=3.12,<3.13", packages={"polars": "==1.2.1"})
class SugrGamesFlow(FlowSpec):
    param_input = Parameter("input", required=True, type=str)
    param_keep = Parameter("keep", default=False)
    param_player_limit = Parameter("player-limit", default=6)
    param_subset = Parameter("subset", default=False)

    @step
    def start(self) -> None:
        import os
        from pathlib import Path

        import polars as pl
        from utilities.hive_dataset import HiveDataset
        from utilities.working_dir import WorkingDirectory

        input_dir = Path(typing.cast(str, self.param_input))

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
            Complexity=pl.String,  # type: ignore [argumentType]
        )

        self.ephemeral = temp.push_segment("ephemeral")
        os.environ["POLARS_TEMP_DIR"] = str(
            self.ephemeral.push_segment(
                "polars",
            ),
        )

        self.source = temp.push_segment("source")
        self.input_expansions_ds = HiveDataset.from_tsv(
            self.source.path,
            input_dir / "expansions.tsv",
        )
        self.input_adversaries_ds = HiveDataset.from_tsv(
            self.source.path,
            input_dir / "adversaries.tsv",
        )
        self.input_escalations_ds = HiveDataset.from_tsv(
            self.source.path,
            input_dir / "escalations.tsv",
        )
        self.input_spirits_ds = HiveDataset.from_tsv(
            self.source.path,
            input_dir / "spirits.tsv",
        )

        self.input_combinations = {
            i: input_dir / "combinations" / f"{i}.parquet" for i in range(1, 7)
        }

        self.next(self.fanout_expansions)

    @step
    def fanout_expansions(self) -> None:
        import polars as pl
        from utilities.hive_dataset import HiveDataset

        from transformations.sugr.expansions import expansions_and_players

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

        self.expansions = expansions_and_players(
            self.input_expansions_ds.read(),
            subset=typing.cast(bool, self.param_subset),
            max_players=typing.cast(int, self.param_player_limit),
        )

        self.next(self.filter_by_expansion, foreach="expansions")

    @step
    def filter_by_expansion(self) -> None:
        from transformations.sugr.adversaries import adversaries_by_expansions
        from transformations.sugr.spirits import spirits_by_expansions

        (self.expansion, self.players) = typing.cast(
            tuple[int, list[int]],
            self.input,
        )

        (adversaries, self.matchups) = adversaries_by_expansions(
            self.expansion,
            self.input_adversaries_ds.read(),
            self.input_escalations_ds.read(),
        )
        self.adversaries_ds.write(adversaries, Expansion=self.expansion)

        self.spirits_ds.write(
            spirits_by_expansions(
                self.expansion,
                self.input_spirits_ds.read(),
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

        self.next(self.fanout_players)

    @step
    def fanout_players(self) -> None:
        self.next(self.generate_combinations, foreach="players")

    @step
    def generate_combinations(self) -> None:
        import polars as pl

        from transformations.sugr.spirits import generate_combinations

        pc = typing.cast(int, self.input)

        print(self.expansion, self.matchup, pc)
        self.combinations_ds.write(
            generate_combinations(
                pc,
                self.matchups_ds.read(
                    Expansion=self.expansion,
                    Matchup=self.matchup,
                ),
                pl.scan_parquet(self.input_combinations[pc]),
            ),
            Expansion=self.expansion,
            Players=pc,
            Matchup=self.matchup,
        )

        self.next(self.collect_players)

    @step
    def collect_players(self, inputs: typing.Any) -> None:
        self.merge_artifacts(
            inputs,
            include=[*__OUTPUT_ARTIFACTS__, *__DATASETS__],
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
        self.next(self.branch_gametypes)

    @step
    def branch_gametypes(self) -> None:
        self.next(self.bucket_horizons, self.bucket_preje)

    @step
    def bucket_horizons(self) -> None:
        from transformations.sugr.expansions import horizons
        from transformations.sugr.games import (
            create_games,
            filter_by_bucket,
            horizons_bucket,
        )

        bucket = horizons_bucket()
        print(str(bucket))
        self.games_ds.write(
            filter_by_bucket(
                bucket,
                create_games(
                    horizons(self.adversaries_ds.read()),
                    horizons(self.combinations_ds.read()),
                ),
            ),
            Difficulty=bucket.difficulty,
            Complexity=bucket.complexity,
        )

        self.next(self.join_gametypes)

    @step
    def bucket_preje(self) -> None:
        from transformations.sugr.expansions import preje
        from transformations.sugr.games import (
            create_games,
            filter_by_bucket,
            preje_buckets,
        )

        games = create_games(
            preje(self.adversaries_ds.read()),
            preje(self.combinations_ds.read()),
        )

        for bucket in preje_buckets(games):
            print(str(bucket))
            self.games_ds.write(
                filter_by_bucket(bucket, games),
                Difficulty=bucket.difficulty,
                Complexity=bucket.complexity,
            )

        self.next(self.join_gametypes)

    @step
    def join_gametypes(self, inputs: typing.Any) -> None:
        self.merge_artifacts(inputs, include=[*__OUTPUT_ARTIFACTS__, *__DATASETS__])
        self.next(self.end)

    @step
    def end(self) -> None:
        if not self.param_keep:
            self.ephemeral.cleanup()


if __name__ == "__main__":
    SugrGamesFlow()
