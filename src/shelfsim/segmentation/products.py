"""Product clusters: the substitution groups for cross-prices (M2) and the nests (nested logit)."""

import numpy as np
import polars as pl
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

FEATURES = ["price_per_oz", "size_oz", "is_private_label", "avg_discount_depth", "promo_frequency"]


def product_features(sales: pl.DataFrame, products: pl.DataFrame) -> pl.DataFrame:
    observed = sales.filter(pl.col("row_status") == "observed")
    any_promo = (
        pl.col("on_price_cut")
        | pl.col("on_feature").fill_null(False)
        | pl.col("on_display").fill_null(False)
    )
    stats = observed.group_by("product_id").agg(
        median_price=pl.col("unit_price").median(),
        avg_discount_depth=pl.col("discount_depth").mean(),
        promo_frequency=any_promo.mean(),
    )
    return (
        products.select("product_id", "name", "manufacturer", "size_oz", "is_private_label")
        .join(stats, on="product_id")
        .with_columns(
            price_per_oz=pl.col("median_price") / pl.col("size_oz"),
            is_private_label=pl.col("is_private_label").cast(pl.Float64),
        )
        .sort("product_id")
    )


def fit_labels(x: np.ndarray, method: str, k: int, seed: int) -> np.ndarray:
    if method == "kmeans":
        return KMeans(n_clusters=k, n_init=20, random_state=seed).fit_predict(x)
    # Diagonal covariance: with 15 to 30 products a full covariance per component is not
    # estimable.
    return GaussianMixture(
        n_components=k, covariance_type="diag", n_init=5, random_state=seed
    ).fit_predict(x)


def silhouette_table(
    x: np.ndarray, k_range: tuple[int, int], min_size: int, seed: int, methods=("kmeans", "gmm")
) -> pl.DataFrame:
    rows = []
    for k in range(k_range[0], min(k_range[1], len(x) - 1) + 1):
        for method in methods:
            labels = fit_labels(x, method, k, seed)
            n_used = len(set(labels))
            sizes = np.bincount(labels)
            rows.append(
                {
                    "method": method,
                    "k": k,
                    "clusters_used": n_used,
                    "smallest_cluster": int(sizes[sizes > 0].min()),
                    "silhouette": float(silhouette_score(x, labels)) if n_used > 1 else np.nan,
                }
            )
    return pl.DataFrame(rows)


def pick(table: pl.DataFrame, min_size: int) -> dict:
    """Best silhouette among solutions where every cluster has at least min_size members,
    because a one-product nest or substitution group has nothing to substitute with."""
    ok = table.filter(
        (pl.col("smallest_cluster") >= min_size) & (pl.col("clusters_used") == pl.col("k"))
    )
    if ok.is_empty():
        ok = table
    return ok.sort("silhouette", descending=True).row(0, named=True)


def name_clusters(features: pl.DataFrame, overrides: dict[int, str]) -> dict[int, str]:
    overall_ppo = features["price_per_oz"].median()
    overall_promo = features["promo_frequency"].median()
    profiles = features.group_by("cluster").agg(
        pl.col("price_per_oz").mean(),
        pl.col("size_oz").mean(),
        pl.col("is_private_label").mean(),
        pl.col("promo_frequency").mean(),
    )
    names = {}
    for row in profiles.iter_rows(named=True):
        ratio = row["price_per_oz"] / overall_ppo
        price = "value" if ratio < 0.85 else "premium" if ratio > 1.15 else "mid-price"
        tier = "private label" if row["is_private_label"] > 0.5 else "national brand"
        size = (
            "large packs"
            if row["size_oz"] >= 19
            else "small packs"
            if row["size_oz"] < 13
            else "standard packs"
        )
        name = f"{price} {tier}, {size}"
        if row["promo_frequency"] > 1.25 * overall_promo:
            name += ", promo-heavy"
        names[row["cluster"]] = name

    seen: dict[str, int] = {}
    for cluster in sorted(names):
        base = names[cluster]
        seen[base] = seen.get(base, 0) + 1
        if seen[base] > 1:
            names[cluster] = f"{base} ({seen[base]})"
    names.update(overrides)
    return names


def cluster_products(
    sales: pl.DataFrame,
    products: pl.DataFrame,
    k_range: tuple[int, int],
    min_size: int,
    seed: int,
    overrides: dict[int, str],
) -> tuple[pl.DataFrame, pl.DataFrame, dict]:
    features = product_features(sales, products)
    x = StandardScaler().fit_transform(features.select(FEATURES).to_numpy())
    table = silhouette_table(x, k_range, min_size, seed)
    choice = pick(table, min_size)
    labels = fit_labels(x, choice["method"], choice["k"], seed)
    # Renumber clusters 0..k-1 by mean price per ounce so the numbering means something.
    features = features.with_columns(raw_cluster=pl.Series(labels))
    order = (
        features.group_by("raw_cluster")
        .agg(pl.col("price_per_oz").mean())
        .sort("price_per_oz")
        .with_row_index("cluster")
    )
    features = features.join(
        order.select("raw_cluster", pl.col("cluster").cast(pl.Int64)), on="raw_cluster"
    ).drop("raw_cluster")
    names = name_clusters(features, overrides)
    features = features.with_columns(cluster_name=pl.col("cluster").replace_strict(names))
    return features, table, choice | {"x": x}
