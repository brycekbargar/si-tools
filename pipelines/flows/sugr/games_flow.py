import gc
import typing

from metaflow import (
    FlowSpec,  # pyright: ignore [reportPrivateImportUsage]
    Parameter,  # pyright: ignore [reportPrivateImportUsage]
    conda_base,  # pyright: ignore [reportAttributeAccessIssue]
    current,  # pyright: ignore [reportPrivateImportUsage]
    step,  # pyright: ignore [reportPrivateImportUsage]
)


@conda_base(python=">=3.12,<3.13", packages={"polars": ">=0.20.2,<1"})
class SugrGamesFlow(FlowSpec):
    param_input = Parameter("input", required=True, type=str)
    param_keep = Parameter("keep", default=False)

    @step
    def start(self) -> None:
        import os
        from pathlib import Path

        from utilities.working_dir import WorkingDirectory

        input_dir = Path(typing.cast(str, self.param_input))
        self.layouts_tsv = input_dir / "layouts.tsv"
        self.expansions_tsv = input_dir / "expansions.tsv"
        self.spirits_tsv = input_dir / "spirits.tsv"
        self.adversaries_tsv = input_dir / "adversaries.tsv"
        self.escalations_tsv = input_dir / "escalations.tsv"

        temp = WorkingDirectory.for_metaflow_run(
            "sugr-games",
            typing.cast(int, current.run_id),
        )
        self.output = temp.push_segment("results")

        self.ephemeral = temp.push_segment("ephemeral")
        os.environ["POLARS_TEMP_DIR"] = str(self.ephemeral.push_segment("polars"))

        self.next(self.fanout_expansions)

    @step
    def fanout_expansions(self) -> None:
        import polars as pl

        expansions = (
            pl.scan_csv(self.expansions_tsv, separator="\t")
            .filter(pl.col("Value").is_in([1, 2, 13, 31]))
            .collect(streaming=True)
            .rows(named=True)
        )

        self.expansion_partitions = [
            (
                typing.cast(int, e["Value"]),
                typing.cast(int, e["Players"]),
                self.ephemeral.push_partitions(("expansion", e["Value"])),
            )
            for e in expansions
        ]

        gc.collect()
        self.next(self.filter_by_expansion, foreach="expansion_partitions")

    @step
    def filter_by_expansion(self) -> None:
        import polars as pl
        from utilities.working_dir import WorkingDirectory

        from transformations.sugr.adversaries import adversaries_by_expansions
        from transformations.sugr.spirits import (
            calculate_matchups,
            spirits_by_expansions,
        )

        (self.expansion, self.max_players, self.expansion_partition) = typing.cast(
            tuple[int, int, WorkingDirectory],
            self.input,
        )

        (adversaries, matchups) = adversaries_by_expansions(
            self.expansion,
            pl.scan_csv(self.adversaries_tsv, separator="\t"),
            pl.scan_csv(self.escalations_tsv, separator="\t"),
        )
        adversaries.sink_parquet(
            self.expansion_partition.file("adversaries.parquet"),
            maintain_order=False,
        )
        del adversaries

        spirits = spirits_by_expansions(
            self.expansion,
            pl.scan_csv(self.spirits_tsv, separator="\t"),
        )
        spirits.sink_parquet(
            self.expansion_partition.file("spirits.parquet"),
            maintain_order=False,
        )

        for matchup in matchups:
            (
                calculate_matchups(
                    matchup,
                    spirits,
                )
            ).sink_parquet(
                self.expansion_partition.file(f"matchup_{matchup}.parquet"),
                maintain_order=False,
            )
        del spirits

        gc.collect()
        self.next(self.fanout_players)

    @step
    def fanout_players(self) -> None:
        self.player_partitions = [
            (
                pc - 1,
                self.expansion_partition.push_partitions(
                    ("players", pc),
                ),
            )
            for pc in range(1, self.max_players + 1)
        ]
        self.next(self.generate_combinations, foreach="player_partitions")

    @step
    def generate_combinations(self) -> None:
        import polars as pl
        from utilities.working_dir import WorkingDirectory

        from transformations.sugr.spirits import generate_combinations

        (pindex, self.player_partition) = typing.cast(
            tuple[int, WorkingDirectory],
            self.input,
        )

        for i in range(pindex):
            if i == 0:
                combinations = generate_combinations(
                    pl.scan_parquet(
                        self.expansion_partition.file("matchup_*.parquet"),
                    ),
                )
                continue

            combinations = generate_combinations(
                pl.scan_parquet(
                    self.expansion_partition.file("matchup_*.parquet"),
                ),
                combinations,
            )

        combinations.sink_parquet(
            self.player_partition.file("combinations.parquet"),
            maintain_order=False,
        )
        del combinations

        gc.collect()
        self.next(self.combine_games)

    @step
    def combine_games(self) -> None:
        import polars as pl

        from transformations.sugr.games import combine

        combine(
            pl.scan_parquet(self.expansion_partition.file("adversaries.parquet")),
            pl.scan_parquet(self.player_partition.file("combinations.parquet")),
        ).sink_parquet(
            self.player_partition.file("games.parquet"),
            maintain_order=False,
        )

        gc.collect()
        self.next(self.collect_players)

    @step
    def collect_players(self, inputs: typing.Any) -> None:
        self.merge_artifacts(
            inputs,
            include=["ephemeral", "output", "expansion_partitions"],
        )
        self.next(self.collect_expansions)

    @step
    def collect_expansions(self, inputs: typing.Any) -> None:
        self.merge_artifacts(
            inputs,
            include=["ephemeral", "output", "expansion_partitions"],
        )
        self.next(self.define_buckets)

    @step
    def define_buckets(self) -> None:
        import polars as pl

        from transformations.sugr.games import define_buckets

        self.buckets = self.ephemeral.push_segment("buckets")

        games_parquet = self.ephemeral.glob_partitions(
            "expansion",
            "players",
        ).file("games.parquet")

        (self.difficulty, self.complexity) = define_buckets(
            pl.scan_parquet(games_parquet),
        )

        gc.collect()
        self.next(self.fanout_buckets)

    @step
    def fanout_buckets(self) -> None:
        self.bucket_partitions = []
        for expansion, max_players, _ in self.expansion_partitions:
            for players in range(1, max_players + 1):
                for d in range(5):
                    for c in range(5):
                        self.bucket_partitions.append(
                            (
                                (d, c),
                                self.ephemeral.push_partitions(
                                    ("expansion", expansion),
                                    ("players", players),
                                ).file("games.parquet"),
                                self.output.push_partitions(
                                    ("expansion", expansion),
                                    ("players", players),
                                    ("difficulty", d),
                                    ("complexity", c),
                                ).file("games.parquet"),
                            ),
                        )

        self.next(self.bucket_games, foreach="bucket_partitions")

    @step
    def bucket_games(self) -> None:
        import polars as pl

        from transformations.sugr.games import filter_by_bucket

        (bucket, source, output) = typing.cast(
            tuple[tuple[int, int], str, str],
            self.input,
        )

        (
            filter_by_bucket(
                bucket,
                self.difficulty,
                self.complexity,
                pl.scan_parquet(source),
            )
        ).sink_parquet(output, maintain_order=False)

        gc.collect()
        self.next(self.collect_buckets)

    @step
    def collect_buckets(self, inputs: typing.Any) -> None:
        self.merge_artifacts(inputs, include=["ephemeral", "output"])
        self.next(self.end)

    @step
    def end(self) -> None:
        import shutil

        if not self.param_keep:
            shutil.rmtree(str(self.ephemeral))


if __name__ == "__main__":
    SugrGamesFlow()
