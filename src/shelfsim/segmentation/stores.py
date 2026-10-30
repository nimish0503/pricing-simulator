"""Store clusters: the groups the virtual-shopper simulator draws a price sensitivity for."""

import numpy as np
import polars as pl
from sklearn.preprocessing import StandardScaler

from shelfsim.segmentation.products import fit_labels, pick, silhouette_table


def store_features(sales: pl.DataFrame, stores: pl.DataFrame) -> pl.DataFrame:
    observed = sales.filter(pl.col("row_status") == "observed")
    chain_price = observed.group_by("product_id", "week_start").agg(
        chain_price=pl.col("unit_price").mean()
    )
    index = (
        observed.join(chain_price, on=["product_id", "week_start"])
        .group_by("store_id")
        .agg(price_index=(pl.col("unit_price") / pl.col("chain_price")).mean())
    )
    weekly = (
        sales.group_by("store_id", "week_start")
        .agg(pl.col("units").sum())
        .group_by("store_id")
        .agg(avg_weekly_units=pl.col("units").mean())
    )
    # Promo responsiveness: mean log ratio of promo-week units to regular-week units, per
    # store-product, averaged over products.
    lift = (
        observed.group_by("store_id", "product_id")
        .agg(
            promo=pl.col("units").filter(pl.col("on_price_cut")).mean(),
            regular=pl.col("units").filter(~pl.col("on_price_cut")).mean(),
        )
        .filter(pl.col("promo").is_not_null() & pl.col("regular").is_not_null())
        .group_by("store_id")
        .agg(promo_lift=(pl.col("promo") / pl.col("regular")).log().mean().exp())
    )
    return (
        stores.select("store_id", pl.col("segment").fill_null("unknown"))
        .join(index, on="store_id")
        .join(weekly, on="store_id")
        .join(lift, on="store_id", how="left")
        .with_columns(pl.col("promo_lift").fill_null(pl.col("promo_lift").median()))
        .sort("store_id")
    )


def cluster_stores(
    sales: pl.DataFrame,
    stores: pl.DataFrame,
    k_range: tuple[int, int],
    min_size: int,
    seed: int,
    overrides: dict[int, str],
) -> tuple[pl.DataFrame, pl.DataFrame, dict]:
    features = store_features(sales, stores)
    numeric = features.select("price_index", "avg_weekly_units", "promo_lift").to_numpy()
    segments = features["segment"].to_dummies().to_numpy()
    if segments.shape[1] > 1:
        numeric = np.hstack([numeric, segments])
    x = StandardScaler().fit_transform(numeric)
    table = silhouette_table(x, k_range, min_size, seed, methods=("kmeans",))
    choice = pick(table, min_size)
    labels = fit_labels(x, "kmeans", choice["k"], seed)
    features = features.with_columns(raw_cluster=pl.Series(labels))
    order = (
        features.group_by("raw_cluster")
        .agg(pl.col("price_index").mean())
        .sort("price_index")
        .with_row_index("cluster")
    )
    features = features.join(
        order.select("raw_cluster", pl.col("cluster").cast(pl.Int64)), on="raw_cluster"
    ).drop("raw_cluster")

    profiles = features.group_by("cluster").agg(
        pl.col("price_index").mean(),
        pl.col("avg_weekly_units").mean(),
        pl.col("promo_lift").mean(),
        pl.col("segment").mode().first(),
    )
    median_units = features["avg_weekly_units"].median()
    names = {}
    for row in profiles.iter_rows(named=True):
        price = (
            "low-price"
            if row["price_index"] < 0.99
            else "high-price"
            if row["price_index"] > 1.01
            else "chain-price"
        )
        volume = (
            "high-volume"
            if row["avg_weekly_units"] > 1.15 * median_units
            else "low-volume"
            if row["avg_weekly_units"] < 0.85 * median_units
            else "mid-volume"
        )
        names[row["cluster"]] = f"{price}, {volume} ({row['segment']} mostly)"
    names.update(overrides)
    features = features.with_columns(cluster_name=pl.col("cluster").replace_strict(names))
    return features, table, choice | {"x": x}
