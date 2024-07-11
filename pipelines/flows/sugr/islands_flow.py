import typing

from metaflow import (
    FlowSpec,  # pyright: ignore [reportPrivateImportUsage]
    Parameter,  # pyright: ignore [reportPrivateImportUsage]
    conda_base,  # pyright: ignore [reportAttributeAccessIssue]
    current,  # pyright: ignore [reportPrivateImportUsage]
    retry,  # pyright: ignore [reportAttributeAccessIssue]
    step,  # pyright: ignore [reportPrivateImportUsage]
)

__OUTPUT_ARTIFACTS__ = ("ephemeral", "islands_ds")


@conda_base(python=">=3.12,<3.13", packages={"polars": ">=0.20.21,<1"})
class SugrIslandsFlow(FlowSpec):
    param_input = Parameter("input", required=True, type=str)
    param_keep = Parameter("keep", default=False)

    @step
    def start(self) -> None:
        import os
        from pathlib import Path

        from utilities.hive_dataset import HiveDataset
        from utilities.working_dir import WorkingDirectory

        self.layouts_tsv = Path(typing.cast(str, self.param_input)) / "layouts.tsv"

        temp = WorkingDirectory.for_metaflow_run(
            "sugr-islands",
            typing.cast(int, current.run_id),
        )
        self.islands_ds = HiveDataset(
            temp.push_segment("results").path,
            "islands",
            "Type",
            "Players",
        )

        self.ephemeral = temp.push_segment("ephemeral")
        os.environ["POLARS_TEMP_DIR"] = str(
            self.ephemeral.push_segment(
                "polars",
            ),
        )

        self.next(self.branch_islandtype)

    @step
    def branch_islandtype(self) -> None:
        self.next(self.branch_loose_board_islands, self.fixed_board_islands)

    @step
    def branch_loose_board_islands(self) -> None:
        self.next(self.fanout_boardcount, self.explode_layouts)

    @step
    def fanout_boardcount(self) -> None:
        self.board_counts = [(4, 4), (6, 6), (8, 6)]
        self.next(self.generate_board_combinations, foreach="board_counts")

    @step
    def generate_board_combinations(self) -> None:
        from utilities.hive_dataset import HiveDataset

        from transformations.sugr.islands import generate_board_combinations

        (board_count, max_players) = typing.cast(
            tuple[typing.Literal[4, 6, 8], int],
            self.input,
        )

        boards_ds = HiveDataset(self.ephemeral.path, "boards", "Type", "Players")
        for pc in list(range(1, max_players + 1)):
            boards_ds.write(
                generate_board_combinations(
                    board_count,
                    pc,
                ),
                Type=f"{board_count}B",
                Players=pc,
            )

        self.next(self.collect_boardcount)

    @step
    def collect_boardcount(self, inputs: typing.Any) -> None:
        self.merge_artifacts(inputs, include=list(__OUTPUT_ARTIFACTS__))
        self.next(self.join_loose_board_islands)

    @retry
    @step
    def explode_layouts(self) -> None:
        import polars as pl
        from utilities.dataset import Dataset

        from transformations.sugr.islands import explode_layouts

        Dataset(self.ephemeral.path, "layouts").write(
            pl.concat(
                explode_layouts(pl.scan_csv(self.layouts_tsv, separator="\t"), pc)
                for pc in range(1, 7)
            ),
        )

        self.next(self.join_loose_board_islands)

    @step
    def join_loose_board_islands(self, inputs: typing.Any) -> None:
        self.merge_artifacts(inputs, include=list(__OUTPUT_ARTIFACTS__))
        self.next(self.generate_loose_islands)

    @step
    def generate_loose_islands(self) -> None:
        from utilities.dataset import Dataset

        from transformations.sugr.islands import generate_loose_islands

        self.islands_ds.write(
            generate_loose_islands(
                Dataset(self.ephemeral.path, "layouts").read(),
                Dataset(self.ephemeral.path, "boards").read(how="diagonal"),
            ),
        )

        self.next(self.join_islandtypes)

    @step
    def fixed_board_islands(self) -> None:
        from transformations.sugr.islands import generate_fixed_islands

        # TODO: Make this data driven?
        for p in range(1, 3 + 1):
            self.islands_ds.write(generate_fixed_islands(p), Type="FB", Players=p)

        self.next(self.join_islandtypes)

    @step
    def join_islandtypes(self, inputs: typing.Any) -> None:
        self.merge_artifacts(inputs, include=list(__OUTPUT_ARTIFACTS__))
        self.next(self.end)

    @step
    def end(self) -> None:
        if not self.param_keep:
            self.ephemeral.cleanup()


if __name__ == "__main__":
    SugrIslandsFlow()
