from metaflow import FlowSpec, step, conda_base, current
from pathlib import Path
import typing
import gc


@conda_base(python=">=3.12,<3.13", packages={"polars": ">=0.20.2,<1"})
class SugrIslandsFlow(FlowSpec):
    @step
    def start(self):
        root = Path("./data")

        self.temp = root / "temp" / str(current.run_id)
        self.output = root / "results" / str(current.run_id)
        Path(self.temp).mkdir(parents=True, exist_ok=True)
        Path(self.output).mkdir(parents=True, exist_ok=True)

        self.layouts_tsv = root / "layouts.tsv"

        self.next(self.branch_islandtype)

    @step
    def branch_islandtype(self):
        self.next(self.loose_board_islands, self.fixed_board_islands)

    @step
    def loose_board_islands(self):
        self.next(self.explode_layouts)

    @step
    def explode_layouts(self):
        import polars as pl
        from transformations.sugr_islands import explode_layouts

        self.layouts_parquet = self.temp / "layouts.parquet"
        (explode_layouts(pl.scan_csv(self.layouts_tsv, separator="\t"))).sink_parquet(
            self.layouts_parquet, maintain_order=False
        )

        gc.collect()
        self.next(self.fanout_boardcount)

    @step
    def fanout_boardcount(self):
        self.board_count = [("4B", 4), ("6B", 6)]
        self.next(self.fanout_players, foreach="board_count")

    @step
    def fanout_players(self):
        (self.island_type, self.max_players) = typing.cast(tuple[str, int], self.input)
        self.board_count = self.max_players
        self.players = range(1, self.max_players + 1)
        self.next(self.generate_board_combinations, foreach="players")

    @step
    def generate_board_combinations(self):
        from transformations.sugr_islands import generate_board_combinations

        self.players = typing.cast(int, self.input)
        self.boards_parquet = (
            self.temp / f"{self.island_type}_{self.players}_boards.parquet"
        )
        (
            generate_board_combinations(
                self.board_count,
                self.players,
            )
        ).sink_parquet(self.boards_parquet, maintain_order=False)

        gc.collect()
        self.next(self.generate_loose_islands)

    @step
    def generate_loose_islands(self):
        from transformations.sugr_islands import generate_loose_islands
        import polars as pl

        islands = generate_loose_islands(
            pl.scan_parquet(self.layouts_parquet),
            pl.scan_parquet(self.boards_parquet),
        )

        self.islands_parquet = (
            self.temp / f"{self.island_type}{self.players:02}_islands.parquet"
        )
        islands.sink_parquet(self.islands_parquet, maintain_order=False)
        self.islands_arrow = (
            self.output / f"{self.island_type}{self.players:02}_islands.arrow"
        )
        islands.sink_ipc(self.islands_arrow, maintain_order=False)

        del islands
        gc.collect()
        self.next(self.count_islands)

    @step
    def count_islands(self):
        import polars as pl

        self.count = (
            pl.scan_parquet(self.islands_parquet)
            .select(pl.count())
            .collect(streaming=True)
            .item()
        )

        gc.collect()
        self.next(self.collect_players)

    @step
    def collect_players(self, inputs):
        self.merge_artifacts(inputs, include=["temp", "output"])

        # do something with counts
        self.next(self.collect_boardcount)

    @step
    def collect_boardcount(self, inputs):
        self.merge_artifacts(inputs, include=["temp", "output"])

        # do something with counts
        self.next(self.join_islandtypes)

    @step
    def fixed_board_islands(self):
        from transformations.sugr_islands import generate_fixed_islands

        # TODO: Make this data driven?
        for p in range(1, 3 + 1):
            islands = generate_fixed_islands(p)

            islands_parquet = self.temp / f"FB{p:02}_islands.parquet"
            islands.sink_parquet(islands_parquet, maintain_order=False)
            islands_arrow = self.temp / f"FB{p:02}_islands.arrow"
            islands.sink_ipc(islands_arrow, maintain_order=False)

            # count = islands.select(pl.count()).collect(streaming=True).item

            del islands
            gc.collect

        # do something with counts
        self.next(self.join_islandtypes)

    @step
    def join_islandtypes(self, inputs):
        self.merge_artifacts(inputs, include=["temp", "output"])

        # do something with counts
        self.next(self.write_stats)

    @step
    def write_stats(self):
        # do something with counts
        self.next(self.end)

    @step
    def end(self):
        pass


if __name__ == "__main__":
    SugrIslandsFlow()
