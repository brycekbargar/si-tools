import polars as pl


class TestLooseIslands:
    def test_generates_combinations(self):
        layouts = pl.LazyFrame(
            {
                "Layout": ["1p1", "3p", "2p", "1p2"],
                "Players": [1, 3, 2, 1],
                "Standard": [True] * 4,
            }
        )

        players = pl.LazyFrame({"Players": [3, 1]})

        from transformations.sugr.islands import generate_loose_islands

        islands = (
            generate_loose_islands(layouts, players).collect().to_dict(as_series=False)
        )

        assert islands == {"Layout": ["1p1", "3p", "1p2"], "Standard": [True] * 3}
