import polars as pl

import transformations.sugr.islands as uut


class TestLooseIslands:
    layouts = pl.LazyFrame(
        {
            "Layout": ["1p1", "1p2", "2p", "3p1", "3p2", "3p3"],
            "Players": [1, 1, 2, 3, 3, 3],
            "Weight": [0.33, 0.66, 1, 0.25, 0.25, 0.5],
            "Standard": [False, True] + [False] * 3 + [True],
        }
    )
    boards = pl.LazyFrame(
        {
            "Boards": ["1b1", "3b", "2b", "1b2"],
            "Players": [1, 3, 2, 1],
        }
    )

    def test_explode_layouts(self):
        layouts = (
            uut.explode_layouts(self.layouts, 3)
            .collect(streaming=True)
            .to_dict(as_series=False)
        )

        assert 42 <= sum([lout == "3p3" for lout in layouts["Layout"]]) <= 58
        assert 17 <= sum([lout == "3p1" for lout in layouts["Layout"]]) <= 33
        assert 17 <= sum([lout == "3p2" for lout in layouts["Layout"]]) <= 33

        assert sorted(layouts.keys()) == sorted(["Layout", "Standard"])

    def test_generate_loose_islands(self):
        islands = (
            uut.generate_loose_islands(
                self.layouts.filter(pl.col("Players") == 1),
                self.boards.filter(pl.col("Players") == 1),
            )
            .collect(streaming=True)
            .to_dict(as_series=False)
        )

        expected = sorted(
            [
                ("1p1", "1b1"),
                ("1p1", "1b2"),
                ("1p2", "1b1"),
                ("1p2", "1b2"),
            ]
        )

        assert (
            sorted(
                [
                    (islands["Layout"][i], islands["Boards"][i])
                    for i in range(len(islands["Layout"]))
                ]
            )
            == expected
        )
        assert islands["Layout"][0] == "1p2"
        assert sorted(islands.keys()) == sorted(["Layout", "Boards", "Standard"])
