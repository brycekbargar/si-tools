from metaflow import FlowSpec, step, conda_base


@conda_base(python=">=3.12,<3.13", packages={"polars": ">=0.20.2,<1"})
class SugrGamesFlow(FlowSpec):
    @step
    def start(self):
        import polars as pl

        expansions = pl.read_csv("./data/expansions.tsv", separator="\t")
        self.expansions = [
            (row["Value"], row["Players"]) for row in expansions.iter_rows(named=True)
        ]
        self.next(self.fanout_expansions, foreach="expansions")

    @step
    def fanout_expansions(self):
        (self.expansion, self.player_count) = self.input
        self.next(self.collect_expansions)

    @step
    def collect_expansions(self, inputs):
        self.collected = [i.expansion for i in inputs]
        self.next(self.end)

    @step
    def end(self):
        print(self.collected)


if __name__ == "__main__":
    SugrGamesFlow()
