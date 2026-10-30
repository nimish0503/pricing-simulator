"""Canonical tables every client is mapped into. See docs/adr/0001-canonical-schema.md.

Onboarding validates against these before writing anything. Code downstream of onboarding
can assume a frame that passed here, whichever client it came from.
"""

import pandera.polars as pa
import polars as pl

_key = pa.Column(pl.String, pa.Check.str_length(min_value=1))
_positive_price = pa.Column(pl.Float64, pa.Check.gt(0))
_flag = pa.Column(pl.Boolean)

sales = pa.DataFrameSchema(
    {
        "client_id": _key,
        "store_id": _key,
        "product_id": _key,
        "week_start": pa.Column(pl.Date),
        "units": pa.Column(pl.Int64, pa.Check.ge(0)),
        "revenue": pa.Column(pl.Float64, pa.Check.ge(0)),
        "unit_price": _positive_price,
        "base_price": _positive_price,
        # 1 - unit_price / base_price. A price above the regular price is not a discount,
        # so onboarding has to decide what to do with it before this check.
        "discount_depth": pa.Column(pl.Float64, pa.Check.in_range(0, 1, include_max=False)),
        "on_feature": _flag,
        "on_display": _flag,
        "on_price_cut": _flag,
        # Only client B reports margin; client A's cost comes from an assumption (ADR 0005).
        "unit_cost": pa.Column(pl.Float64, pa.Check.ge(0), nullable=True),
    },
    unique=["client_id", "store_id", "product_id", "week_start"],
    strict=True,
    name="sales",
)

products = pa.DataFrameSchema(
    {
        "client_id": _key,
        "product_id": _key,
        "name": pa.Column(pl.String),
        "brand": pa.Column(pl.String),
        "manufacturer": pa.Column(pl.String, nullable=True),
        "category": pa.Column(pl.String),
        "size_oz": pa.Column(pl.Float64, pa.Check.gt(0)),
        "is_private_label": _flag,
    },
    unique=["client_id", "product_id"],
    strict=True,
    name="products",
)

# Not strict: clients bring different demographic columns and we keep them.
stores = pa.DataFrameSchema(
    {
        "client_id": _key,
        "store_id": _key,
        "segment": pa.Column(pl.String, nullable=True),
        "region": pa.Column(pl.String, nullable=True),
    },
    unique=["client_id", "store_id"],
    strict=False,
    name="stores",
)
