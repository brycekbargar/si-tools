import polars as pl

import transformations.sugr.islands as uut


class TestLooseIslands:
    layouts = pl.LazyFrame(
        {
            "Layout": ["1p1", "3p", "2p", "1p2"],
            "Players": [1, 3, 2, 1],
            "Standard": [False] * 2 + [True, False],
        }
    )
    boards = pl.LazyFrame(
        {
            "Boards": ["1b1", "3b", "2b", "1b2"],
            "Players": [1, 3, 2, 1],
        }
    )

    def test_generates_combinations(self):
        boards = self.boards
        boards = boards.filter(pl.col("Players") != 2)
        islands = (
            uut.generate_loose_islands(self.layouts, boards)
            .collect(streaming=True)
            .to_dict(as_series=False)
        )

        expected = sorted(
            [
                ("1p1", "1b1"),
                ("1p1", "1b2"),
                ("1p2", "1b1"),
                ("1p2", "1b2"),
                ("3p", "3b"),
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

    def test_standard_first(self):
        islands = (
            uut.generate_loose_islands(self.layouts, self.boards)
            .collect(streaming=True)
            .to_dict(as_series=False)
        )

        assert islands["Layout"][0] == "2p"
