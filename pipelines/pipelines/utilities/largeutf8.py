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
# %mamba install pyarrow --yes --quiet

# %%
from pathlib import Path


# %%
def make_longutf8_shorter(
    in_parquet: Path, out_arrow: Path, compression: str = "uncompressed"
):
    import pyarrow as pa
    import pyarrow.feather as pf
    import pyarrow.parquet as pq

    schema = pq.read_schema(in_parquet)
    schema = pa.schema(
        [
            pa.field(
                f.name,
                pa.string() if f.type == pa.large_string() else f.type,
                nullable=f.nullable,
            )
            for f in [
                schema.field(i)
                for i in [schema.get_field_index(n) for n in schema.names]
            ]
        ]
    )
    table = pq.read_table(in_parquet, schema=schema)

    pf.write_feather(table, out_arrow, compression=compression)


# %%
if hasattr(__builtins__, "__IPYTHON__"):
    make_longutf8_shorter(
        "./data/temp/1703783237129398/01010000.parquet", "./test.arrow"
    )

# %%
