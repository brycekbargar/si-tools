from metaflow import FlowSpec, step, conda_base, current


@conda_base(python=">=3.12,<3.13", packages={"polars": ">=0.20.2,<1"})
class SugrGamesFlow(FlowSpec):
    @step
    def start(self):
        import polars as pl

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
        self.players = list(range(1, self.max_players + 1))

        self.next(self.generate_combinations, foreach="players")

    @step
    def generate_combinations(self):
        from transformations.sugr_spirits import generate_combinations
        import polars as pl
        import typing

        combinations = self.adversaries.join(
            pl.concat(
                [
                    generate_combinations(typing.cast(int, self.input), m)
                    for m in self.matchups
                ]
            ),
            on="Matchup",
        )

        combinations.sink_parquet(
            f"./data/results/{current.run_id}/{self.expansions:2}{self.players:2}.parquet",
            maintain_order=False,
            statistics=True,
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
