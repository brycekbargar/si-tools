import typing

from metaflow import (
    FlowSpec,  # pyright: ignore [reportPrivateImportUsage]
    Parameter,  # pyright: ignore [reportPrivateImportUsage]
    conda_base,  # pyright: ignore [reportAttributeAccessIssue]
    current,  # pyright: ignore [reportPrivateImportUsage]
    step,  # pyright: ignore [reportPrivateImportUsage]
    trigger_on_finish,  # pyright: ignore [reportAttributeAccessIssue]
)


@conda_base(python=">=3.12,<3.13", packages={"polars": "==1.2.1", "pyarrow": "17.0.0"})
@trigger_on_finish(flows=["SugrIslandsFlow", "SugrGamesFlow"])
class SiteSugrFlow(FlowSpec):
    param_output = Parameter("output", required=True, type=str)

    @step
    def start(self) -> None:
        import os
        import shutil
        from pathlib import Path

        from utilities.working_dir import WorkingDirectory

        temp = WorkingDirectory.for_metaflow_run(
            "site-sugr",
            typing.cast(int, current.run_id),
        )

        self.output = Path(typing.cast(str, self.param_output))
        if (io := self.output / "islands").exists():
            shutil.rmtree(io)
        if (go := self.output / "games").exists():
            shutil.rmtree(go)

        self.ephemeral = temp.push_segment("ephemeral")
        os.environ["POLARS_TEMP_DIR"] = str(
            self.ephemeral.push_segment(
                "polars",
            ),
        )

        self.next(self.branch_flowtypes)

    @step
    def branch_flowtypes(self) -> None:
        if typing.TYPE_CHECKING:
            from utilities.hive_dataset import HiveDataset

        self.islands_ds: HiveDataset = current.trigger[  # pyright: ignore [reportAttributeAccessIssue]
            "SugrIslandsFlow"
        ].data.islands_ds
        self.games_ds: HiveDataset = current.trigger["SugrGamesFlow"].data.games_ds  # pyright: ignore [reportAttributeAccessIssue]

        self.next(self.fanout_islands, self.fanout_games)

    @step
    def fanout_islands(self) -> None:
        self.partitions = self.islands_ds.partitions()
        self.next(self.package_islands, foreach="partitions")

    @step
    def package_islands(self) -> None:
        from pathlib import Path

        import pyarrow.feather as pf

        from transformations.site.package import batch, drop_nulls

        partition = typing.cast(dict[str, typing.Any], self.input)
        path = (
            self.output
            / "islands"
            / Path(*[f"{k}={v}" for (k, v) in partition.items()])
        )
        path.mkdir(mode=0o755, parents=True, exist_ok=True)

        end = 0
        for (start, e), part in batch(drop_nulls(self.islands_ds.read(**partition))):
            pf.write_feather(
                part.collect(streaming=True).to_arrow(),
                path / f"{start}.feather",
                compression="uncompressed",
            )
            end = e

        print("[Island Partition] ", partition, f": {end} total rows")

        self.next(self.collect_islands)

    @step
    def collect_islands(self, inputs: typing.Any) -> None:
        self.merge_artifacts(inputs, include=["ephemeral"])
        self.next(self.join_flowtypes)

    @step
    def fanout_games(self) -> None:
        self.partitions = self.games_ds.partitions()
        self.next(self.package_games, foreach="partitions")

    @step
    def package_games(self) -> None:
        from pathlib import Path

        import pyarrow.feather as pf

        from transformations.site.package import batch, drop_nulls

        partition = typing.cast(dict[str, typing.Any], self.input)
        path = (
            self.output / "games" / Path(*[f"{k}={v}" for (k, v) in partition.items()])
        )
        path.mkdir(mode=0o755, parents=True, exist_ok=True)

        end = 0
        for (start, e), part in batch(drop_nulls(self.games_ds.read(**partition))):
            pf.write_feather(
                part.collect(streaming=True).to_arrow(),
                path / f"{start}.feather",
                compression="uncompressed",
            )
            end = e

        print("[Games Partition] ", partition, f": {end} total rows")

        self.next(self.collect_games)

    @step
    def collect_games(self, inputs: typing.Any) -> None:
        self.merge_artifacts(inputs, include=["ephemeral"])
        self.next(self.join_flowtypes)

    @step
    def join_flowtypes(self, inputs: typing.Any) -> None:
        self.merge_artifacts(inputs, include=["ephemeral"])
        self.next(self.end)

    @step
    def end(self) -> None:
        self.ephemeral.cleanup()


if __name__ == "__main__":
    SiteSugrFlow()
