from metaflow import FlowSpec, step, conda_base


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
        self.expansions = list(
            self.expansions_tsv.collect().select(["Value", "Players"]).iter_rows()
        )

        self.next(self.filter_by_expansions, foreach="expansions")

    @step
    def filter_by_expansions(self):
        from transformations.sugr_spirits import spirits_by_expansions
        from transformations.sugr_adversaries import adversaries_by_expansions
        import typing

        (self.expansions, self.max_players) = typing.cast(tuple[int, str], self.input)

        self.spirits = spirits_by_expansions(self.expansions, self.spirits_tsv)
        self.spirits = adversaries_by_expansions(
            self.expansions, self.adversaries_tsv, self.escalations_tsv
        )

        self.next(self.fanout_matchups)

    @step
    def fanout_matchups(self):
        # Fanout more
        self.next(self.collect_expansions)

    @step
    def collect_expansions(self, inputs):
        self.next(self.end)

    @step
    def end(self):
        pass


if __name__ == "__main__":
    SugrGamesFlow()
