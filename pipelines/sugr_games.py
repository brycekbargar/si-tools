from metaflow import FlowSpec, step, conda_base, current
from pathlib import Path
import typing
import gc


@conda_base(python=">=3.12,<3.13", packages={"polars": ">=0.20.2,<1"})
class SugrGamesFlow(FlowSpec):
    @step
    def start(self):
        root = Path("./data")

        self.temp = root / "temp" / str(current.run_id)
        self.output = root / "results" / str(current.run_id)
        Path(self.temp).mkdir(parents=True, exist_ok=True)
        Path(self.output).mkdir(parents=True, exist_ok=True)

        self.expansions_tsv = root / "expansions.tsv"
        self.spirits_tsv = root / "spirits.tsv"
        self.adversaries_tsv = root / "adversaries.tsv"
        self.escalations_tsv = root / "escalations.tsv"

        self.next(self.fanout_expansions)

    @step
    def fanout_expansions(self):
        import polars as pl

        self.expansions = (
            pl.scan_csv(self.expansions_tsv, separator="\t")
            .filter(pl.col("Value").is_in([1, 2, 13, 31]))
            .collect(streaming=True)
            .rows(named=True)
        )

        gc.collect()
        self.next(self.filter_by_expansions, foreach="expansions")

    @step
    def filter_by_expansions(self):
        import polars as pl
        from operator import itemgetter
        from transformations.sugr_spirits import spirits_by_expansions
        from transformations.sugr_adversaries import adversaries_by_expansions

        input = itemgetter("Value", "Players")(
            typing.cast(dict[str, typing.Any], self.input)
        )
        (self.expansions, self.max_players) = typing.cast(tuple[int, int], input)

        self.spirits_parquet = self.temp / f"{self.expansion:02}_spirits.parquet"
        (
            spirits_by_expansions(
                self.expansions, pl.scan_csv(self.spirits_tsv, separator="\t")
            )
        ).sink_parquet(self.spirits_parquet, maintain_order=False)

        self.adversaries_parquet = (
            self.temp / f"{self.expansion:02}_adversaries.parquet"
        )
        (adversaries, self.matchups) = adversaries_by_expansions(
            self.expansions,
            pl.scan_csv(self.adversaries_tsv, separator="\t"),
            pl.scan_csv(self.escalations_tsv, separator="\t"),
        )
        adversaries.sink_parquet(self.adversaries_parquet)
        del adversaries

        gc.collect()
        self.next(self.fanout_matchups)

    @step
    def fanout_matchups(self):
        self.next(self.calculate_matchups, foreach="matchups")

    @step
    def calculate_matchups(self):
        import polars as pl
        from transformations.sugr_spirits import calculate_matchups

        self.matchup = typing.cast(str, self.input)

        self.spirits_matchups_parquet = (
            self.temp / f"{self.expansion:02}_{self.matchup}_spirits.parquet"
        )
        (
            calculate_matchups(
                self.matchup,
                pl.scan_parquet(self.spirits_parquet),
            )
        ).sink_parquet(
            self.spirits_matchups_parquet,
            maintain_order=False,
        )

        gc.collect()
        self.next(self.generate_combinations)

    @step
    def generate_combinations(self):
        from transformations.sugr_spirits import generate_combinations
        import polars as pl

        self.combinations_parquet = [
            self.temp / f"{self.expansion:02}_{self.matchup}_01.parquet"
        ]
        (
            generate_combinations(
                self.matchup,
                1,
                pl.scan_parquet(self.spirits_matchups_parquet),
            )
        ).sink_parquet(self.combinations_parquet[0], maintain_order=False)
        gc.collect()

        for players in range(2, self.max_players + 1):
            self.combinations_parquet.append(
                self.temp / f"{self.expansion:02}_{self.matchup}_{players:02}.parquet"
            )
            (
                generate_combinations(
                    self.matchup,
                    players,
                    pl.scan_parquet(self.spirits_matchups_parquet),
                    pl.scan_parquet(self.combinations_parquet[-2]),
                )
            ).sink_parquet(self.combinations_parquet[-1], maintain_order=False)
            gc.collect()

        self.next(self.collect_matchups)

    @step
    def collect_matchups(self, inputs):
        self.combinations_parquet = {}
        for p in range(len(inputs[0].combinations)):
            self.combinations_parquet[p + 1] = [
                i.combinations_parquet[p] for i in inputs
            ]

        self.players = self.combinations_parquet.keys()
        self.next(self.fanout_players)

    @step
    def fanout_players(self):
        self.next(self.combine_games, foreach="players")

    @step
    def combine_games(self):
        from transformations.sugr_games import combine
        import polars as pl

        self.players = typing.cast(int, self.input)

        self.games_parquet = (
            self.temp / f"{self.expansions:02}_{self.players:02}_games.parquet"
        )
        combine(
            pl.scan_parquet(self.adversaries_parquet),
            pl.scan_parquet(self.combinations_parquet[self.players]),
        ).sink_parquet(self.games_parquet, maintain_order=False)
        gc.collect()

        self.next(self.collect_players)

    @step
    def collect_players(self, inputs):
        self.games_parquet_by_players = {i.players: i.game_parquet for i in inputs}
        self.games_parquet = [i.games_parquet for i in inputs]
        self.next(self.collect_expansions)

    @step
    def collect_expansions(self, inputs):
        self.games_parquet_by_expansions_and_players = {
            i.expansions: i.games_parquet_by_players for i in inputs
        }
        self.games_parquet = [
            g for games in [i.games_parquet for i in inputs] for g in games
        ]
        self.next(self.define_buckets)

    @step
    def define_buckets(self):
        from transformations.sugr_games import define_buckets
        import polars as pl

        self.difficulty_parquet = self.temp / "buckets_difficulty.parquet"
        self.complexity_parquet = self.temp / "buckets_complexity.parquet"

        (difficulty, complexity) = define_buckets(pl.scan_parquet(self.games_parquet))
        difficulty.sink_parquet(self.difficulty_parquet, maintain_order=False)
        complexity.sink_parquet(self.complexity_parquet, maintain_order=False)

        del difficulty, complexity
        gc.collect()

        self.next(self.fanout_buckets)

    @step
    def fanout_buckets(self):
        self.buckets = []
        for exp in typing.cast(list[dict[str, typing.Any]], self.expansions):
            for players in range(1, exp["Players"] + 1):
                for d in range(5):
                    for c in range(4):
                        self.buckets.append(
                            (typing.cast(int, exp["Value"]), players, d, c)
                        )

        self.next(self.bucket_games, foreach="buckets")

    @step
    def bucket_games(self):
        from transformations.sugr_games import filter_by_bucket
        import polars as pl

        self.bucket = typing.cast(tuple[int, int, int, int], self.input)
        (expansions, players, difficulty, complexity) = self.bucket

        games = filter_by_bucket(
            (difficulty, complexity),
            pl.scan_parquet(self.difficulty_parquet),
            pl.scan_parquet(self.complexity_parquet),
            pl.scan_parquet(
                typing.cast(
                    list[Path],
                    self.games_parquet_by_expansions_and_players[expansions][players],
                )
            ),
        )

        self.games_parquet = (
            self.temp
            / f"{expansions:02}{players:02}{difficulty:02}{complexity:02}.parquet"
        )
        games.sink_parquet(self.games_parquet, maintain_order=False)
        games_arrow = (
            self.output
            / f"{expansions:02}{players:02}{difficulty:02}{complexity:02}.arrow"
        )
        games.sink_ipc(games_arrow, maintain_order=False)

        del games
        gc.collect()

        self.next(self.count_games)

    @step
    def count_games(self):
        import polars as pl

        self.count = (
            pl.scan_parquet(self.games_parquet)
            .select(pl.count())
            .collect(streaming=True)
            .item()
        )
        gc.collect()

        self.next(self.collect_buckets)

    @step
    def collect_buckets(self, inputs):
        self.stats = [list(i.bucket + (i.count,)) for i in inputs]
        self.next(self.write_stats)

    @step
    def write_stats(self):
        import polars as pl

        schema = [
            "Expansions",
            "Players",
            "Difficulty",
            "Complexity",
            "Count",
        ]
        stats = pl.DataFrame(typing.cast(list[int], self.stats), schema=schema).sort(
            schema[:-1]
        )
        stats.write_csv(self.temp / "stats.tsv", separator="\t")
        stats.write_json(self.output / "stats.json", row_oriented=True)
        del stats
        gc.collect()

        self.next(self.end)

    @step
    def end(self):
        pass


if __name__ == "__main__":
    SugrGamesFlow()
