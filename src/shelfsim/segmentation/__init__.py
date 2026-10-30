"""Product and store clustering on the training period, written to outputs/<client>/."""

from datetime import datetime

import polars as pl
from jinja2 import Environment, PackageLoader, select_autoescape
from sklearn.decomposition import PCA

from shelfsim.config import ClientConfig, ModelingConfig
from shelfsim.io import output_dir, read_table
from shelfsim.report.charts import PALETTE, figure, to_base64
from shelfsim.segmentation.products import cluster_products
from shelfsim.segmentation.stores import cluster_stores
from shelfsim.splits import final_fold


def run(cfg: ClientConfig, mcfg: ModelingConfig) -> None:
    seg = mcfg.segmentation
    sales = read_table(cfg.client_id, "sales")
    train = final_fold(sales, mcfg.split).train(sales)
    products, product_table, product_choice = cluster_products(
        train,
        read_table(cfg.client_id, "products"),
        seg.product_k_range,
        seg.min_cluster_size,
        seg.seed,
        cfg.product_cluster_names,
    )
    stores, store_table, store_choice = cluster_stores(
        train,
        read_table(cfg.client_id, "stores"),
        seg.store_k_range,
        seg.min_cluster_size,
        seg.seed,
        cfg.store_cluster_names,
    )

    out = output_dir(cfg.client_id)
    products.write_csv(out / "product_clusters.csv")
    stores.write_csv(out / "store_clusters.csv")
    product_table.write_csv(out / "silhouette_products.csv")
    store_table.write_csv(out / "silhouette_stores.csv")

    env = Environment(
        loader=PackageLoader("shelfsim.report", "templates"), autoescape=select_autoescape()
    )
    html = env.get_template("clusters.html.j2").render(
        client_name=cfg.display_name,
        generated=datetime.now().strftime("%Y-%m-%d %H:%M"),
        period=(train["week_start"].min(), train["week_start"].max()),
        product_choice={k: v for k, v in product_choice.items() if k != "x"},
        store_choice={k: v for k, v in store_choice.items() if k != "x"},
        product_table=product_table.to_dicts(),
        store_table=store_table.to_dicts(),
        product_profile=_profile(
            products,
            [
                "price_per_oz",
                "size_oz",
                "is_private_label",
                "avg_discount_depth",
                "promo_frequency",
            ],
        ),
        store_profile=_profile(stores, ["price_index", "avg_weekly_units", "promo_lift"]),
        products=products.sort("cluster", "price_per_oz").to_dicts(),
        product_pca=_pca_plot(product_choice["x"], products, "Products"),
        store_pca=_pca_plot(store_choice["x"], stores, "Stores"),
    )
    (out / "clusters.html").write_text(html)


def _profile(frame: pl.DataFrame, columns: list[str]) -> list[dict]:
    return (
        frame.group_by("cluster", "cluster_name")
        .agg(pl.len().alias("members"), *(pl.col(c).mean() for c in columns))
        .sort("cluster")
        .to_dicts()
    )


def _pca_plot(x, frame: pl.DataFrame, title: str) -> str:
    pca = PCA(n_components=2).fit(x)
    points = pca.transform(x)
    fig, ax = figure(7, 4.6)
    names = dict(frame.select("cluster", "cluster_name").unique().iter_rows())
    for cluster in sorted(names):
        mask = (frame["cluster"] == cluster).to_numpy()
        ax.scatter(
            points[mask, 0],
            points[mask, 1],
            s=40,
            color=PALETTE[cluster % len(PALETTE)],
            label=f"{cluster}: {names[cluster]}",
        )
    ratio = pca.explained_variance_ratio_
    ax.set_xlabel(f"PC1 ({ratio[0]:.0%} of variance)")
    ax.set_ylabel(f"PC2 ({ratio[1]:.0%} of variance)")
    ax.set_title(f"{title}: first two principal components of the standardised features")
    ax.legend(fontsize=8, frameon=False, loc="best")
    return to_base64(fig)
