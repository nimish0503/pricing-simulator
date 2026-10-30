"""Data-quality report per client: outputs/<client>/data_quality.html.

The log is filled while onboarding runs, so the row counts are the real ones from each step.
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import polars as pl
from jinja2 import Environment, PackageLoader, select_autoescape


@dataclass
class QualityLog:
    steps: list[dict] = field(default_factory=list)
    dropped: list[pl.DataFrame] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def step(self, name: str, rows: int, detail: str = "") -> None:
        self.steps.append({"step": name, "rows": rows, "detail": detail})

    def drop_products(self, frame: pl.DataFrame, reason: str) -> None:
        if frame.height:
            columns = [c for c in ("product_id", "name") if c in frame.columns]
            self.dropped.append(frame.select(columns).with_columns(reason=pl.lit(reason)))

    def dropped_products(self) -> pl.DataFrame:
        if not self.dropped:
            return pl.DataFrame(
                schema={"product_id": pl.String, "name": pl.String, "reason": pl.String}
            )
        return pl.concat(self.dropped, how="diagonal").unique("product_id", keep="first")


def product_profile(sales: pl.DataFrame, products: pl.DataFrame) -> pl.DataFrame:
    observed = pl.col("row_status") == "observed"
    profile = sales.group_by("product_id").agg(
        store_weeks=pl.len(),
        zero_sales_share=(pl.col("row_status") == "zero_sales").mean(),
        units=pl.col("units").sum(),
        revenue=pl.col("revenue").sum(),
        price_p10=pl.col("unit_price").filter(observed).quantile(0.10),
        price_median=pl.col("unit_price").filter(observed).median(),
        price_p90=pl.col("unit_price").filter(observed).quantile(0.90),
        base_price_median=pl.col("base_price").median(),
        price_cut_share=pl.col("on_price_cut").filter(observed).mean(),
        feature_share=pl.col("on_feature").filter(observed).mean(),
        display_share=pl.col("on_display").filter(observed).mean(),
        mean_depth_on_cut=pl.col("discount_depth").filter(observed & pl.col("on_price_cut")).mean(),
        outliers=pl.col("is_outlier").sum(),
    )
    return (
        products.select("product_id", "name", "manufacturer", "size_oz", "is_private_label")
        .join(profile, on="product_id")
        .sort("revenue", descending=True)
    )


def base_price_check(frame: pl.DataFrame) -> dict | None:
    """Agreement between our estimated base price and a base price the client supplied."""
    if "given_base_price" not in frame.columns:
        return None
    rows = frame.filter(
        (pl.col("row_status") == "observed") & pl.col("given_base_price").is_not_null()
    )
    gap = (pl.col("base_price") / pl.col("given_base_price").cast(pl.Float64) - 1).abs()
    stats = rows.select(
        within_2pct=(gap <= 0.02).mean(),
        within_5pct=(gap <= 0.05).mean(),
        median_gap=gap.median(),
        n=pl.len(),
    ).row(0, named=True)
    return stats


def write_report(
    path: Path,
    client_name: str,
    log: QualityLog,
    sales: pl.DataFrame,
    products: pl.DataFrame,
    stores: pl.DataFrame,
    base_check: dict | None,
) -> None:
    env = Environment(
        loader=PackageLoader("shelfsim.report", "templates"), autoescape=select_autoescape()
    )
    profile = product_profile(sales, products)
    dropped = log.dropped_products()
    status = sales.group_by("row_status").len().sort("row_status")
    summary = {
        "weeks": sales["week_start"].n_unique(),
        "first_week": sales["week_start"].min(),
        "last_week": sales["week_start"].max(),
        "stores": sales["store_id"].n_unique(),
        "products": sales["product_id"].n_unique(),
        "rows": sales.height,
        "zero_sales_share": (sales["row_status"] == "zero_sales").mean(),
        "outliers": int(sales["is_outlier"].sum()),
        "price_above_base_rows": int(sales["price_above_base"].sum()),
        "stores_without_segment": int(stores["segment"].null_count()),
    }
    html = env.get_template("data_quality.html.j2").render(
        client_name=client_name,
        generated=datetime.now().strftime("%Y-%m-%d %H:%M"),
        summary=summary,
        steps=log.steps,
        status=status.to_dicts(),
        profile=profile.to_dicts(),
        dropped_counts=dropped.group_by("reason").len().sort("len", descending=True).to_dicts(),
        dropped=dropped.head(60).to_dicts(),
        notes=log.notes,
        base_check=base_check,
    )
    path.write_text(html)
