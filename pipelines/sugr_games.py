from metaflow import FlowSpec, step, conda_base, current


@conda_base(python=">=3.12,<3.13", packages={"polars": ">=0.20.2,<1"})
class SugrGamesFlow(FlowSpec):
    @step
    def start(self):
        import polars as pl
        from pathlib import Path

        self.concat_hack = f"./data/temp/{current.run_id}"
        Path(self.concat_hack).mkdir(parents=True, exist_ok=True)
        self.output_parquet = f"./data/results/{current.run_id}"
        Path(self.output_parquet).mkdir(parents=True, exist_ok=True)

        self.expansions_tsv = pl.scan_csv("./data/expansions.tsv", separator="\t")
        self.spirits_tsv = pl.scan_csv("./data/spirits.tsv", separator="\t")
        self.adversaries_tsv = pl.scan_csv("./data/adversaries.tsv", separator="\t")
        self.escalations_tsv = pl.scan_csv("./data/escalations.tsv", separator="\t")

        self.next(self.fanout_expansions)

    @step
    def fanout_expansions(self):
        self.expansions = (
            self.expansions_tsv.collect().select(["Value", "Players"]).rows()
        )

        self.next(self.filter_by_expansions, foreach="expansions")

    @step
    def filter_by_expansions(self):
        from transformations.sugr_spirits import spirits_by_expansions
        from transformations.sugr_adversaries import adversaries_by_expansions
        import typing

        (self.expansions, self.max_players) = typing.cast(tuple[int, int], self.input)

        self.spirits = spirits_by_expansions(self.expansions, self.spirits_tsv)
        self.adversaries = adversaries_by_expansions(
            self.expansions, self.adversaries_tsv, self.escalations_tsv
        )

        self.next(self.calculate_matchups)

    @step
    def calculate_matchups(self):
        from transformations.sugr_adversaries import unique_matchups
        from transformations.sugr_spirits import calculate_matchups

        self.matchups = [
            calculate_matchups(m, self.spirits)
            for m in [r[0] for r in unique_matchups(self.adversaries).collect().rows()]
        ]

        self.next(self.fanout_players)

    @step
    def fanout_players(self):
        self.players = list(range(1, min(self.max_players, 3) + 1))

        self.next(self.generate_combinations, foreach="players")

    @step
    def generate_combinations(self):
        from transformations.sugr_games import combine
        import typing

        print(f"expansions:{self.expansions}, players:{self.input}")
        combinations = combine(
            typing.cast(int, self.input),
            self.adversaries,
            self.matchups,
            hack_concat_file=f"{self.concat_hack}/{current.task_id}",
        )
        combinations.sink_parquet(
            f"{self.output_parquet}/{self.expansions:02}{self.input:02}.parquet",
            maintain_order=False,
        )

        self.next(self.collect_players)

    @step
    def collect_players(self, inputs):
        self.next(self.collect_expansions)

    @step
    def collect_expansions(self, inputs):
        self.next(self.end)

    @step
    def end(self):
        pass


if __name__ == "__main__":
    SugrGamesFlow()
