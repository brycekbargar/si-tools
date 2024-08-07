import polars as pl


def test_prejebuckets() -> None:
    from transformations.sugr.games import preje_buckets as uut

    games = pl.DataFrame(
        {
            "Expansion": [12, 15, 15, 15, 15, 15, 32, 2],
            "Players": [2, 2, 2, 2, 1, 4, 2, 3],
            "Difficulty": [60000, 1, 2, 3, 9000, 10000, 50000, 1000],
            "Spirit_0": [
                "Not Birb",
                "Not Birb",
                "Not Birb",
                "Finder of Paths Unseen",
                "Not Birb",
                "Not Birb",
                "N/A",
                "N/A",
            ],
            "Spirit_1": [
                "Not Birb",
                None,
                "Finder of Paths Unseen",
                "Not Birb",
                None,
                "Not Birb",
                "N/A",
                "N/A",
            ],
            "Spirit_2": ["Finder of Paths Unseen"] + [None] * 7,
            "Spirit_3": [None] * 8,
        },
    )

    buckets = list(uut(games.lazy()))
    assert len(buckets) == 6

    (zero_birb,) = (b.expr for b in buckets if b.difficulty == 0 and b.complexity == 1)
    (zero_nobirb,) = (
        b.expr for b in buckets if b.difficulty == 0 and b.complexity == 0
    )
    assert games.filter(zero_birb).height == 0
    assert games.filter(zero_nobirb).height == 1

    (one_birb,) = (b.expr for b in buckets if b.difficulty == 1 and b.complexity == 1)
    (one_nobirb,) = (b.expr for b in buckets if b.difficulty == 1 and b.complexity == 0)
    assert games.filter(one_birb).height == 1
    assert games.filter(one_nobirb).height == 0

    (two_birb,) = (b.expr for b in buckets if b.difficulty == 2 and b.complexity == 1)
    (two_nobirb,) = (b.expr for b in buckets if b.difficulty == 2 and b.complexity == 0)
    assert games.filter(two_birb).height == 2
    assert games.filter(two_nobirb).height == 2
