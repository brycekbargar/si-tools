from metaflow import FlowSpec, step, conda_base, current
from pathlib import Path
import typing

# start
# branch_islandtype
#   loose_board_islands
#       explode_layouts -> loose_board_layouts
#       branch_boardcount
#           four_board_islands
#               fanout_loose_players
#                   generate_board_combinations -> loose_board_combinations
#                   generate_loose_islands -> loose_board_islands
#                   count_islands -> inline
#           six_board_islands
#               fanout_loose_players
#                   generate_board_combinations
#                   generate_loose_islands
#                   count_islands
#               collect_loose_players
#       join_boardcount
#   fixed_board_islands
#       fanout_fixed_players
#           generate_fixed_islands -> fixed_board_islands
#           count_islands -> inline
# join_islandtype
# write_stats -> inline
# end


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
        # layouts_parquet
        self.next(self.fanout_boardcount)

    @step
    def fanout_boardcount(self):
        self.board_count = [("4B", 4), ("6B", 6)]
        self.next(self.fanout_players, foreach="board_count")

    @step
    def fanout_players(self):
        (self.island_type, self.max_players) = typing.cast(tuple[str, int], self.input)
        self.players = range(1, self.max_players + 1)
        self.next(self.generate_board_combinations, foreach="players")

    @step
    def generate_board_combinations(self):
        # {self.island_type}_{self.players}_boards.parquet
        self.next(self.generate_loose_islands)

    @step
    def generate_loose_islands(self):
        # {self.island_type}{self.players}.parquet
        self.next(self.count_islands)

    @step
    def count_islands(self):
        self.next(self.collect_players)

    @step
    def collect_players(self, inputs):
        self.next(self.join_boardcount)

    @step
    def join_boardcount(self, inputs):
        self.next(self.join_islandtypes)

    @step
    def fixed_board_islands(self):
        # FB{i}.parquet
        # count
        self.next(self.join_islandtypes)

    @step
    def join_islandtypes(self, inputs):
        self.next(self.write_stats)

    @step
    def write_stats(self):
        self.next(self.end)

    @step
    def end(self):
        pass


if __name__ == "__main__":
    SugrIslandsFlow()
