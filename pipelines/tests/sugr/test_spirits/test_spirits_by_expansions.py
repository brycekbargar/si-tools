import polars as pl

import transformations.sugr.spirits as uut


class TestSpiritsByExpansions:
    def test_spirits_by_expansion(self) -> None:
        spirits = pl.LazyFrame(
            {
                "Name": ["S1", "S2", "S2", "S3", "S3", "S4"],
                "Aspect": [None, None, "2A1", None, "3A1", None],
                "Expansions": [1, 1, 4, 1, 8, 16],
                "Complexity": ["Moderate"] * 6,
            },
        )

        results = (
            uut.spirits_by_expansions(1 | 8, spirits)
            .collect(streaming=True)
            .to_dict(as_series=False)
        )

        # filters by expansion
        assert "S4" not in results["Name"]
        # only makes base aspect for spirit with multiple aspects
        assert results["Aspect"][results["Name"].index("S1")] is None
        assert results["Aspect"][results["Name"].index("S2")] is None
        # makes base aspect when appicable
        assert sorted(
            [
                a
                for a in results["Aspect"]
                if results["Name"][results["Aspect"].index(a)] == "S3"
            ],
        ) == sorted(["Base", "3A1"])

        # converts complexity to something usable
        for c in results["Complexity"]:
            assert isinstance(c, int)
            assert c > 0
