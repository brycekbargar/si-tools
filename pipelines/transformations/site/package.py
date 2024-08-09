"""Provides operations on LazyFrames useful for packaging them for the web."""

import gc
import typing

import polars as pl


def drop_nulls(frame: pl.LazyFrame) -> pl.LazyFrame:
    """Removes any columns only containing null values."""
    nulls = (
        frame.clone().select(pl.all().is_null().all()).unpivot().filter(pl.col("value"))
    )
    null_cols = [r[0] for r in nulls.collect(streaming=True).rows()]
    del nulls
    gc.collect()

    return frame.clone().drop(null_cols)


def batch(
    frame: pl.LazyFrame,
    size: int = 10_000,
) -> typing.Iterator[tuple[tuple[int, int], pl.LazyFrame]]:
    """Batches the frame, 'size' rows at a time."""
    all_records = frame.clone().with_row_index()

    last_idx = 0
    more = True
    while more:
        next_idx = last_idx + size
        batch = all_records.clone().filter(
            pl.Expr.and_(
                pl.col("index").ge(last_idx),
                pl.col("index").lt(next_idx),
            ),
        )

        rem = batch.select("index").collect(streaming=True).height
        yield ((last_idx, last_idx + rem), batch.clone().drop("index"))

        last_idx = next_idx
        more = size == rem
        del batch
        gc.collect()

    del all_records
    gc.collect()


def sample(
    frame: pl.LazyFrame,
    samples: int = 100_000,
) -> pl.LazyFrame:
    """Reduces the frame to the length of samples."""
    if samples in [0, 1, 2]:
        return frame.head(samples)

    rows = frame.select(pl.len()).collect(streaming=True).item(0, 0)
    gc.collect()

    if rows <= samples:
        return frame

    # We want to randomly sample the whole frame
    # instead of just the back part after an offset
    # This gets close with a chunk skipped in the middle but it will be smallish.
    (n_extra, extra) = divmod(rows, samples)
    (n_sample, offset) = divmod(rows - extra, samples - (extra // n_extra))

    return pl.concat(
        [
            frame.clone().head(extra).gather_every(n_extra),
            frame.clone().gather_every(n_sample, extra + offset),
        ],
    ).head(samples)
