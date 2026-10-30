"""Raw client files in, validated canonical tables out."""

import logging
from pathlib import Path

import duckdb
import polars as pl

from shelfsim import schema
from shelfsim.config import ROOT, ClientConfig, ModelingConfig
from shelfsim.io import DUCKDB_PATH, canonical_dir, output_dir
from shelfsim.onboarding import steps
from shelfsim.onboarding.adapters import read_source
from shelfsim.onboarding.quality import QualityLog, base_price_check, write_report

log = logging.getLogger(__name__)

SALES_COLUMNS = list(schema.sales.columns)


def onboard(
    cfg: ClientConfig,
    mcfg: ModelingConfig,
    root: Path = ROOT,
    out_dir: Path | None = None,
    canon_dir: Path | None = None,
    db_path: Path | None = DUCKDB_PATH,
) -> dict[str, pl.DataFrame]:
    quality = QualityLog()
    rules = mcfg.onboarding

    raw_sales = read_source(cfg.sales, cfg.raw_dir, root)
    quality.step("read raw sales", raw_sales.height, cfg.sales.path)
    raw_products = read_source(cfg.products, cfg.raw_dir, root)
    raw_stores = read_source(cfg.stores, cfg.raw_dir, root)
    quality.step("read raw products", raw_products.height, cfg.products.path)

    products = steps.build_products(raw_products, cfg, quality)
    stores = steps.build_stores(raw_stores, cfg)

    sales = steps.unit_fixes(raw_sales, cfg)
    if cfg.row_filters:
        quality.step(
            "unit fixes",
            sales.height,
            f"{(~sales['valid']).sum():,} rows fail {' and '.join(cfg.row_filters)} and are treated "
            "as no data",
        )
    if "qty" in sales.columns:
        bundles = sales.filter(pl.col("qty") > 1).height
        quality.notes.append(
            f"{bundles:,} raw rows are bundle prices (qty > 1). Unit price = price / qty."
        )

    sales = steps.to_calendar(sales, cfg)
    sales, products = steps.select_window_and_products(sales, products, cfg.selection, quality)
    quality.step(
        "analysis window and product selection",
        sales.height,
        f"last {cfg.selection.last_weeks} weeks, {products.height} products",
    )

    sales = steps.fill_missing_weeks(sales, rules, quality)
    sales = steps.add_base_price(sales, rules)
    quality.step(
        "base price",
        sales.height,
        f"trailing {rules.base_price_window_weeks}-week max of non-promo prices",
    )
    sales = steps.promo_flags(sales, rules)
    quality.step(
        "promo flags",
        sales.height,
        f"price cut = promo code or depth >= {rules.price_cut_min_depth:.0%}",
    )
    sales = steps.flag_outliers(sales, rules)
    quality.step(
        "outliers",
        sales.height,
        f"{int(sales['is_outlier'].sum()):,} regular-price weeks flagged above median + {rules.outlier_mad_k:g} MAD",
    )
    sales = steps.add_unit_cost(sales, products, cfg)

    for signal in ("feature", "display"):
        if getattr(cfg.promo, signal) is None:
            quality.notes.append(f"No {signal} data in this client's files. on_{signal} is null.")
    gaps = (
        sales.select(pl.col("week_start").unique().sort())
        .with_columns(gap=pl.col("week_start").diff().dt.total_days() / 7 - 1)
        .filter(pl.col("gap") > 0)
    )
    for row in gaps.iter_rows(named=True):
        weeks = int(row["gap"])
        quality.notes.append(
            f"No store reported for {weeks} week{'s' if weeks > 1 else ''} before {row['week_start']}."
        )

    canonical_sales = sales.with_columns(client_id=pl.lit(cfg.client_id)).select(SALES_COLUMNS)
    schema.sales.validate(canonical_sales)
    schema.products.validate(products)
    stores = stores.join(canonical_sales.select("store_id").unique(), on="store_id")
    schema.stores.validate(stores)
    quality.step("validated and written", canonical_sales.height, "pandera schemas passed")

    tables = {"sales": canonical_sales, "products": products, "stores": stores}
    _write(tables, cfg.client_id, canon_dir or canonical_dir(cfg.client_id), db_path)
    write_report(
        (out_dir or output_dir(cfg.client_id)) / "data_quality.html",
        cfg.display_name,
        quality,
        sales,
        products,
        stores,
        base_price_check(sales),
    )
    return tables


def _write(
    tables: dict[str, pl.DataFrame], client_id: str, canon: Path, db_path: Path | None
) -> None:
    canon.mkdir(parents=True, exist_ok=True)
    for name, frame in tables.items():
        frame.write_parquet(canon / f"{name}.parquet")
    if db_path is None:
        return
    db_path.parent.mkdir(parents=True, exist_ok=True)
    core = {
        "sales": tables["sales"],
        "products": tables["products"],
        "stores": tables["stores"].select(list(schema.stores.columns)),
        # Stores carry client-specific demographic columns, so they go in a long table.
        "store_attributes": tables["stores"]
        .drop("client_id", "segment", "region")
        .unpivot(index="store_id", variable_name="attribute", value_name="value")
        .with_columns(client_id=pl.lit(client_id), value=pl.col("value").cast(pl.Float64)),
    }
    with duckdb.connect(str(db_path)) as con:
        for name, frame in core.items():
            arrow = frame.to_arrow()
            con.register("incoming", arrow)
            con.execute(f"CREATE TABLE IF NOT EXISTS {name} AS SELECT * FROM incoming LIMIT 0")
            con.execute(f"DELETE FROM {name} WHERE client_id = ?", [client_id])
            con.execute(f"INSERT INTO {name} BY NAME SELECT * FROM incoming")
            con.unregister("incoming")
