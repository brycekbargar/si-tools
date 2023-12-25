# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
# %conda install polars=="0.19.13" --yes

# %%
import polars as pl

if hasattr(__builtins__, "__IPYTHON__"):
    from rich import inspect, print

    inspect("ruff/isort strip out rich")
    print("unless it is used")

    pl.show_versions()


# %%
def combine(
    players: int,
    adversaries: pl.LazyFrame,
    matchups: list[pl.LazyFrame],
    hack_concat_file: str | None = None,
) -> pl.LazyFrame:
    """Combines spirits and adversaries to create the complete set of games."""

    import polars as pl

    from .sugr_spirits import generate_combinations

    # pl.concat doesn't work with streaming
    if hack_concat_file:
        for i, m in enumerate(matchups):
            generate_combinations(players, m).sink_parquet(
                f"{hack_concat_file}_{i}.parquet",
                maintain_order=False,
            )
        matchups = pl.scan_parquet(f"{hack_concat_file}_*.parquet")
    else:
        matchups = pl.concat([generate_combinations(players, m) for m in matchups])

    return (
        adversaries.join(
            matchups,
            on="Matchup",
        )
        .with_columns(
            [
                pl.col("Difficulty").add(pl.col("Difficulty_right")),
                pl.col("Complexity").add(pl.col("Complexity_right")),
            ]
        )
        .drop("Difficulty_right", "Complexity_right", "Matchup")
    )


# %%
if hasattr(__builtins__, "__IPYTHON__"):
    import polars as pl
    import sugr_adversaries as sa
    import sugr_spirits as ss

    spirits = ss.spirits_by_expansions(
        31, pl.scan_csv("../data/spirits.tsv", separator="\t")
    )
    # spirits.sink_parquet("test_spirits.parquet", maintain_order=False)
    adversaries = sa.adversaries_by_expansions(
        31,
        pl.scan_csv("../data/adversaries.tsv", separator="\t"),
        pl.scan_csv("../data/escalations.tsv", separator="\t"),
    )
    # adversaries.sink_parquet("test_adversaries.parquet", maintain_order=False)

    matchups = [
        ss.calculate_matchups(m[0], spirits)
        for m in sa.unique_matchups(adversaries).collect().rows()
    ]
    # for i, m in enumerate(matchups):
    #  matchups[i].sink_parquet(f"test_matchups_{i}.parquet", maintain_order=False)
    # pl.concat(matchups, how="align", parallel=False).sink_parquet("test_matchups.parqet", maintain_order=False)
    # matchups[0].merge_sorted(matchups[1], key="Name").sink_parquet("test_matchups.parqet", maintain_order=False)

    games = combine(5, adversaries, matchups, hack_concat_file="./hack")
    games.sink_parquet("test_games.parquet", maintain_order=False)
    # games.show_graph(streaming=True)
    # games.collect(streaming=True)

# %%
if hasattr(__builtins__, "__IPYTHON__"):
    print(pl.scan_parquet("./test_games.parquet").tail(15).collect())

# %%

# %%
