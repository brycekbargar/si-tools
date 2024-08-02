import polars as pl

from transformations.sugr.adversaries import adversaries_by_expansions as uut


class TestAdversariesByExpansions:
    adversaries = pl.LazyFrame(
        {
            "Name": [
                "A1",
                "A1",
                "A2",
                "The Kingdom of France (Plantation Colony)",
                "The Kingdom of France (Plantation Colony)",
                "The Kingdom of France (Plantation Colony)",
                "A4",
                "A5",
            ],
            "Expansion": [1, 1, 1, 4, 4, 4, 8, 16],
            "Level": [1, 2, 1, 4, 5, 6, 1, 1],
            "Matchup": [
                "Tier",
                "Tier",
                "Scotland",
                "France",
                "France",
                "France",
                "England",
                "Mines",
            ],
            "Difficulty": [0] * 8,
            "Complexity": ["Moderate"] * 8,
        },
    )
    escalations = pl.LazyFrame(
        {
            "Escalation": ["E1", "E2", "E3", "E4"],
            "Expansion": [1, 4, 8, 16],
        },
    )

    def test_filters_and_collects(self) -> None:
        (filtered, collected) = uut(
            1 | 4 | 16,
            self.adversaries,
            pl.LazyFrame(),
        )

        expected = self.adversaries.collect().to_dict(as_series=False)
        filtered = (filtered.collect(streaming=True).to_dict(as_series=False))[
            "Adversary"
        ]

        assert sorted(filtered) == sorted([n for n in expected["Name"] if n != "A4"])
        assert sorted(collected) == sorted(
            {n for n in expected["Matchup"] if n != "England"},
        )

    def test_adds_escalations(self) -> None:
        (horizons, _) = uut(
            2,
            self.adversaries,
            self.escalations,
        )

        (escalations, _) = uut(
            1 | 16,
            self.adversaries,
            self.escalations,
        )

        (no_escalations, _) = uut(
            1 | 4 | 8 | 16,
            self.adversaries,
            self.escalations,
        )

        assert sorted(
            {
                e
                for e in horizons.collect(streaming=True).to_dict(
                    as_series=False,
                )["Escalation"]
                if e is not None
            },
        ) == sorted(["E1", "E2", "E3", "E4"])

        assert sorted(
            {
                e
                for e in escalations.collect(streaming=True).to_dict(
                    as_series=False,
                )["Escalation"]
                if e is not None
            },
            # Only "new" escalations based on what expansions are included
        ) == sorted(["E2", "E3"])

        assert "Escalation" not in no_escalations.collect(streaming=True).to_dict(
            as_series=False,
        )

    def test_fuck_france(self) -> None:
        (adversaries, _) = uut(
            1 | 4 | 8 | 16,
            self.adversaries,
            pl.LazyFrame(),
        )

        france = (
            adversaries.filter(pl.col("Matchup").eq(pl.lit("France")))
            .collect()
            .to_dict(as_series=False)
        )

        assert len(france["Adversary"]) == 1
        assert france["Level"][0] == 4
