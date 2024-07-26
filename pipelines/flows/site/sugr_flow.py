import typing

from metaflow import (
    FlowSpec,  # pyright: ignore [reportPrivateImportUsage]
    Parameter,  # pyright: ignore [reportPrivateImportUsage]
    conda_base,  # pyright: ignore [reportAttributeAccessIssue]
    current,  # pyright: ignore [reportPrivateImportUsage]
    step,  # pyright: ignore [reportPrivateImportUsage]
    trigger_on_finish,  # pyright: ignore [reportAttributeAccessIssue]
)


@conda_base(python=">=3.12,<3.13", packages={"polars": "==1.1.0"})
@trigger_on_finish(flows=["SugrIslandsFlow", "SugrGamesFlow"])
class SiteSugrFlow(FlowSpec):
    param_output = Parameter("output", required=True, type=str)

    @step
    def start(self) -> None:
        import os
        from pathlib import Path

        from utilities.working_dir import WorkingDirectory

        temp = WorkingDirectory.for_metaflow_run(
            "site-sugr",
            typing.cast(int, current.run_id),
        )

        self.output = Path(typing.cast(str, self.param_output))

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

        self.next(self.package_islands, self.package_games)

    @step
    def package_islands(self) -> None:
        from pathlib import Path

        for partition in self.islands_ds.partitions():
            path = (
                self.output
                / "islands"
                / Path(*[f"{k}={v}" for (k, v) in partition.items()])
            )
            path.mkdir(mode=0o755, parents=True, exist_ok=True)

            self.islands_ds.read(**partition).sink_ipc(
                path / "0.feather",
                compression=None,
                maintain_order=False,
            )

        self.next(self.join_flowtypes)

    @step
    def package_games(self) -> None:
        from pathlib import Path

        for partition in self.games_ds.partitions():
            path = (
                self.output
                / "games"
                / Path(*[f"{k}={v}" for (k, v) in partition.items()])
            )
            path.mkdir(mode=0o755, parents=True, exist_ok=True)

            self.games_ds.read(**partition).sink_ipc(
                path / "0.feather",
                compression=None,
                maintain_order=False,
            )

        self.next(self.join_flowtypes)

    @step
    def join_flowtypes(self, inputs: typing.Any) -> None:
        self.merge_artifacts(inputs, include=["ephemeral"])
        self.next(self.end)

    @step
    def end(self) -> None:
        if not self.param_keep:
            self.ephemeral.cleanup()


if __name__ == "__main__":
    SiteSugrFlow()
