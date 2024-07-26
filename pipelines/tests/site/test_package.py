import polars as pl


def test_drop_nulls() -> None:
    from transformations.site.package import drop_nulls as uut

    frame = pl.LazyFrame(
        {
            "no-nulls": [1, 2, 3, 4],
            "all-nulls": [None, None, None, None],
            "some-nulls": [1, 2, None, 4],
            "another-all-nulls": [None, None, None, None],
        },
    )

    results = uut(frame)
    assert results.collect(streaming=True).schema.names() == ["no-nulls", "some-nulls"]


def test_batch() -> None:
    from transformations.site.package import batch as uut

    frame = pl.LazyFrame(
        [*range(100)],
        schema="a",
        orient="row",
    )

    batches = []
    for i, ((s, e), f) in enumerate(uut(frame, 15)):
        batches.append(s)

        if i < 6:
            assert f.collect(streaming=True).height == 15
        else:
            assert f.collect(streaming=True).height == 10
            assert e == 100

    assert batches == [0, 15, 30, 45, 60, 75, 90]
