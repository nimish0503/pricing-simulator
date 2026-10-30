"""Export what the app needs to outputs/deploy/<client>/: model outputs and aggregates only.

Neither dataset may be redistributed, so nothing at store x product x week grain leaves here.
"""

import json
import shutil

import polars as pl

from shelfsim.config import ClientConfig
from shelfsim.io import OUTPUTS_DIR, output_dir, read_table
from shelfsim.onboarding.quality import product_profile

DEPLOY_DIR = OUTPUTS_DIR / "deploy"
COPY = [
    "elasticities.csv",
    "cross_elasticities.csv",
    "logit_elasticities.csv",
    "logit_cross_elasticities.csv",
    "product_clusters.csv",
    "promo_plan.csv",
    "backtest_summary.csv",
    "backtest.md",
    "simulator.json",
    "optimizer.json",
    "choice_models.json",
]
RAW_GRAIN = {"store_id", "product_id", "week_start"}


def export(cfg: ClientConfig) -> None:
    source, target = output_dir(cfg.client_id), DEPLOY_DIR / cfg.client_id
    target.mkdir(parents=True, exist_ok=True)
    for name in COPY:
        if (source / name).exists():
            shutil.copy(source / name, target / name)

    sales = read_table(cfg.client_id, "sales")
    products = read_table(cfg.client_id, "products")
    weekly = (
        sales.group_by("week_start")
        .agg(
            units=pl.col("units").sum(),
            revenue=pl.col("revenue").sum(),
            promo_unit_share=pl.col("units").filter(pl.col("on_price_cut")).sum()
            / pl.col("units").sum(),
            stores=pl.col("store_id").n_unique(),
        )
        .sort("week_start")
    )
    weekly.write_csv(target / "weekly_category.csv")
    # Chain-level product x week: summed over stores, so it is an aggregate, not raw rows.
    (
        sales.group_by("product_id", "week_start")
        .agg(
            units=pl.col("units").sum(),
            revenue=pl.col("revenue").sum(),
            shelf_price=pl.col("unit_price").median(),
            regular_price=pl.col("base_price").median(),
            stores_on_price_cut=pl.col("on_price_cut").mean(),
            stores_on_feature=pl.col("on_feature").mean(),
            stores_on_display=pl.col("on_display").mean(),
        )
        .sort("product_id", "week_start")
        .write_csv(target / "product_weekly.csv")
    )
    product_profile(sales, products).write_csv(target / "product_profile.csv")
    (target / "summary.json").write_text(
        json.dumps(
            {
                "client_id": cfg.client_id,
                "display_name": cfg.display_name,
                "first_week": str(sales["week_start"].min()),
                "last_week": str(sales["week_start"].max()),
                "stores": sales["store_id"].n_unique(),
                "products": products.height,
                "weeks": sales["week_start"].n_unique(),
                "rows": sales.height,
                "zero_sales_share": float((sales["row_status"] == "zero_sales").mean()),
                "outlier_share": float(sales["is_outlier"].mean()),
                "has_feature": bool(sales["on_feature"].any()),
                "has_display": bool(sales["on_display"].any()),
            },
            indent=1,
        )
    )
    check(target)


def check(directory) -> None:
    """Fail if any exported table sits at the raw store x product x week grain."""
    for path in directory.glob("*.csv"):
        columns = set(pl.read_csv(path, n_rows=1).columns)
        if columns >= RAW_GRAIN:
            raise ValueError(f"{path} has store x product x week rows; it must not be deployed")
