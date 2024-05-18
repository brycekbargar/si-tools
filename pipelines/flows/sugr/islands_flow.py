import gc
import typing

from metaflow import (
    FlowSpec,  # pyright: ignore [reportPrivateImportUsage]
    Parameter,  # pyright: ignore [reportPrivateImportUsage]
    conda_base,  # pyright: ignore [reportAttributeAccessIssue]
    current,  # pyright: ignore [reportPrivateImportUsage]
    retry,  # pyright: ignore [reportAttributeAccessIssue]
    step,  # pyright: ignore [reportPrivateImportUsage]
)


@conda_base(python=">=3.12,<3.13", packages={"polars": ">=0.20.21,<1"})
class SugrIslandsFlow(FlowSpec):
    param_input = Parameter("input", required=True, type=str)
    param_keep = Parameter("keep", default=False)

    @step
    def start(self) -> None:
        import os
        from pathlib import Path

        from utilities.working_dir import WorkingDirectory

        self.layouts_tsv = Path(typing.cast(str, self.param_input)) / "layouts.tsv"

        temp = WorkingDirectory.for_metaflow_run(
            "sugr-islands",
            typing.cast(int, current.run_id),
        )
        self.output = temp.push_segment("results")

        self.ephemeral = temp.push_segment("ephemeral")
        os.environ["POLARS_TEMP_DIR"] = self.ephemeral.push_segment(
            "polars",
        ).directory()

        self.next(self.branch_islandtype)

    @step
    def branch_islandtype(self) -> None:
        self.next(self.loose_board_islands, self.fixed_board_islands)

    @step
    def loose_board_islands(self) -> None:
        self.next(self.fanout_boardcount)

    @step
    def fanout_boardcount(self) -> None:
        self.board_counts = [("4B", 4), ("6B", 6)]
        self.next(self.fanout_players, foreach="board_counts")

    @step
    def fanout_players(self) -> None:
        (self.island_type, self.max_players) = typing.cast(
            tuple[str, typing.Literal[4, 6]],
            self.input,
        )
        self.board_count: typing.Literal[4, 6] = self.max_players
        self.player_partitions = [
            (
                pc,
                self.ephemeral.push_partitions(
                    ("island_type", self.island_type),
                    ("players", pc),
                ),
            )
            for pc in range(1, self.max_players + 1)
        ]
        self.next(self.explode_layouts, foreach="player_partitions")

    @retry
    @step
    def explode_layouts(self) -> None:
        import polars as pl
        from utilities.working_dir import WorkingDirectory

        from transformations.sugr.islands import explode_layouts

        (self.players, self.partition) = typing.cast(
            tuple[int, WorkingDirectory],
            self.input,
        )

        (
            explode_layouts(pl.scan_csv(self.layouts_tsv, separator="\t"), self.players)
        ).sink_parquet(self.partition.file("layouts.parquet"), maintain_order=False)

        gc.collect()
        self.next(self.generate_board_combinations)

    @step
    def generate_board_combinations(self) -> None:
        from transformations.sugr.islands import generate_board_combinations

        (
            generate_board_combinations(
                self.board_count,
                self.players,
            )
        ).sink_parquet(self.partition.file("boards.parquet"), maintain_order=False)

        gc.collect()
        self.next(self.generate_loose_islands)

    @step
    def generate_loose_islands(self) -> None:
        import polars as pl

        from transformations.sugr.islands import generate_loose_islands

        (
            generate_loose_islands(
                pl.scan_parquet(self.partition.file("layouts.parquet")),
                pl.scan_parquet(self.partition.file("boards.parquet")),
            ).sink_parquet(
                self.output.file(f"{self.island_type}{self.players:02}.parquet"),
            )
        )

        gc.collect()
        self.next(self.collect_players)

    @step
    def collect_players(self, inputs: typing.Any) -> None:
        self.merge_artifacts(inputs, include=["ephemeral", "output"])
        self.next(self.collect_boardcount)

    @step
    def collect_boardcount(self, inputs: typing.Any) -> None:
        self.merge_artifacts(inputs, include=["ephemeral", "output"])
        self.next(self.join_islandtypes)

    @step
    def fixed_board_islands(self) -> None:
        from transformations.sugr.islands import generate_fixed_islands

        # TODO: Make this data driven?
        for p in range(1, 3 + 1):
            (
                generate_fixed_islands(p).sink_parquet(
                    self.output.file(f"FB{p:02}_islands.parquet"),
                )
            )

            gc.collect()

        self.next(self.join_islandtypes)

    @step
    def join_islandtypes(self, inputs: typing.Any) -> None:
        self.merge_artifacts(inputs, include=["ephemeral", "output"])
        self.next(self.end)

    @step
    def end(self) -> None:
        import shutil

        if not self.param_keep:
            shutil.rmtree(str(self.ephemeral))


if __name__ == "__main__":
    SugrIslandsFlow()
