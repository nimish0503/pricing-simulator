"""Model panel built from the canonical sales table, shared by every demand model."""

import numpy as np
import polars as pl

KEYS = ["store_id", "product_id"]


def fourier_columns(terms: int) -> list[str]:
    return [f"{fn}{k}" for k in range(1, terms + 1) for fn in ("sin", "cos")]


def build_panel(
    sales: pl.DataFrame, clusters: pl.DataFrame, fourier_terms: int, lags: list[int]
) -> pl.DataFrame:
    """One row per store x product x week with every feature any model uses.

    ln_cross_price is the log of the mean unit price of the *other* products in the same
    product cluster, same store and week. Lags are on calendar weeks, so a missing week gives a
    null lag rather than silently shifting to an older week.
    """
    frame = sales.join(clusters.select("product_id", "cluster"), on="product_id").with_columns(
        feature=pl.col("on_feature").fill_null(False).cast(pl.Float64),
        display=pl.col("on_display").fill_null(False).cast(pl.Float64),
        ln_base_price=pl.col("base_price").log(),
        ln_units=pl.when(pl.col("units") > 0).then(pl.col("units").cast(pl.Float64).log()),
    )
    group = ["store_id", "week_start", "cluster"]
    frame = frame.with_columns(
        others_sum=pl.col("unit_price").sum().over(group) - pl.col("unit_price"),
        others_n=pl.len().over(group) - 1,
    ).with_columns(
        ln_cross_price=pl.when(pl.col("others_n") > 0)
        .then((pl.col("others_sum") / pl.col("others_n")).log())
        .otherwise(None)
    )

    year_fraction = pl.col("week_start").dt.ordinal_day().cast(pl.Float64) / 365.25
    frame = frame.with_columns(
        *(
            (getattr(np, fn)(2 * np.pi * k * year_fraction)).alias(f"{fn}{k}")
            for k in range(1, fourier_terms + 1)
            for fn in ("sin", "cos")
        )
    )

    history = frame.select(*KEYS, "week_start", ln1p_units=pl.col("units").cast(pl.Float64).log1p())
    for lag in lags:
        shifted = history.with_columns(pl.col("week_start") + pl.duration(weeks=lag)).rename(
            {"ln1p_units": f"lag{lag}"}
        )
        frame = frame.join(shifted, on=[*KEYS, "week_start"], how="left")
    return frame.drop("others_sum", "others_n").sort(*KEYS, "week_start")


def fit_rows(frame: pl.DataFrame, exclude_outliers: bool) -> pl.DataFrame:
    """Rows the log models fit on: weeks that sold something, outliers optionally out.

    Zero-sales weeks have no log, and they're under 1% of rows in both clients.
    """
    keep = pl.col("ln_units").is_not_null()
    if exclude_outliers:
        keep &= ~pl.col("is_outlier")
    return frame.filter(keep)


def varying(frame: pl.DataFrame, columns: list[str]) -> list[str]:
    """Columns with any variation. A client that doesn't report display has a constant 0."""
    return [c for c in columns if frame[c].drop_nulls().n_unique() > 1]
