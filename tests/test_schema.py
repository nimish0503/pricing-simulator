from datetime import date

import pandera.errors
import polars as pl
import pytest

from shelfsim import schema


def five_sales_rows() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "client_id": ["client_a"] * 5,
            "store_id": ["367", "367", "367", "389", "389"],
            "product_id": ["1111009477"] * 5,
            "week_start": [
                date(2009, 1, 12),
                date(2009, 1, 19),
                date(2009, 1, 26),
                date(2009, 1, 12),
                date(2009, 1, 19),
            ],
            "units": [13, 0, 25, 8, 9],
            "revenue": [17.42, 0.0, 26.25, 10.72, 12.06],
            "unit_price": [1.34, 1.34, 1.05, 1.34, 1.34],
            "base_price": [1.34, 1.34, 1.34, 1.34, 1.34],
            "discount_depth": [0.0, 0.0, 1 - 1.05 / 1.34, 0.0, 0.0],
            "on_feature": [False, False, True, False, False],
            "on_display": [False, False, True, False, False],
            "on_price_cut": [False, False, True, False, False],
            "unit_cost": [None, None, None, None, None],
        },
        schema_overrides={"unit_cost": pl.Float64},
    )


def test_valid_sales_passes():
    schema.sales.validate(five_sales_rows())


def test_negative_units_rejected():
    bad = five_sales_rows().with_columns(
        pl.when(pl.int_range(pl.len()) == 2).then(-3).otherwise(pl.col("units")).alias("units")
    )
    with pytest.raises(pandera.errors.SchemaError, match="units"):
        schema.sales.validate(bad)


def test_duplicate_store_product_week_rejected():
    rows = five_sales_rows()
    with pytest.raises(pandera.errors.SchemaError):
        schema.sales.validate(pl.concat([rows, rows.head(1)]))


def test_extra_sales_column_rejected():
    with pytest.raises(pandera.errors.SchemaError):
        schema.sales.validate(five_sales_rows().with_columns(pl.lit(1).alias("move")))


def test_products_and_stores():
    products = pl.DataFrame(
        {
            "client_id": ["client_b", "client_b"],
            "product_id": ["3000006", "4116709428"],
            "name": ["KELLOGG CORN FLAKES", "DOM CORN FLAKES"],
            "brand": ["Kellogg's Corn Flakes", "Dominick's"],
            "manufacturer": ["Kellogg", None],
            "category": ["cereal", "cereal"],
            "size_oz": [18.0, 18.0],
            "is_private_label": [False, True],
        }
    )
    schema.products.validate(products)

    stores = pl.DataFrame(
        {
            "client_id": ["client_b"],
            "store_id": ["2"],
            "segment": [None],
            "region": ["chicago"],
            "income": [10.55],
        },
        schema_overrides={"segment": pl.String},
    )
    schema.stores.validate(stores)

    with pytest.raises(pandera.errors.SchemaError, match="size_oz"):
        schema.products.validate(products.with_columns(pl.lit(0.0).alias("size_oz")))
