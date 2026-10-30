"""Fit the choice models, build the simulator, run the promo planner and the optimizer."""

import json
import logging
from datetime import timedelta

import numpy as np
import polars as pl

from shelfsim.choice.logit import LogitModel, fit_logit, market_sizes
from shelfsim.choice.simulator import ClusterParams, Simulator
from shelfsim.config import ClientConfig, ModelingConfig
from shelfsim.demand import load_panel
from shelfsim.io import output_dir, read_output, read_table
from shelfsim.optimize import build_rules, optimize
from shelfsim.promo import plan
from shelfsim.splits import final_fold

log = logging.getLogger(__name__)

CURRENT_WEEKS = 13


def current_products(sales: pl.DataFrame, client: str) -> pl.DataFrame:
    """Today's regular price and unit cost: medians over the last 13 weeks of data."""
    start = sales["week_start"].max() - timedelta(weeks=CURRENT_WEEKS - 1)
    recent = sales.filter(pl.col("week_start") >= start)
    prices = recent.group_by("product_id").agg(
        base_price=pl.col("base_price").median(), unit_cost=pl.col("unit_cost").median()
    )
    clusters = read_output(client, "product_clusters.csv").select(
        pl.col("product_id").cast(pl.String), "cluster", "cluster_name"
    )
    return (
        read_table(client, "products")
        .select("product_id", "name", "manufacturer", "is_private_label", "size_oz")
        .join(clusters, on="product_id")
        .join(prices, on="product_id")
        .sort("product_id")
    )


def nests_of(panel: pl.DataFrame) -> dict[str, int]:
    return dict(panel.select("product_id", "cluster").unique().iter_rows())


def fit_models(
    train: pl.DataFrame, mcfg: ModelingConfig
) -> tuple[LogitModel, LogitModel, pl.DataFrame]:
    sizes = market_sizes(train, mcfg.choice.market_peak_multiple)
    plain = fit_logit(train, sizes, fourier_terms=mcfg.demand.fourier_terms)
    nested = fit_logit(train, sizes, nests=nests_of(train), fourier_terms=mcfg.demand.fourier_terms)
    return plain, nested, sizes


def build_simulator(
    train: pl.DataFrame,
    sizes: pl.DataFrame,
    products: pl.DataFrame,
    store_clusters: pl.DataFrame,
    mcfg: ModelingConfig,
) -> Simulator:
    """One plain logit per store cluster gives alpha_cluster and the product utilities there."""
    ids = products["product_id"].to_list()
    params = []
    for (cluster, name), stores in store_clusters.group_by(
        "cluster", "cluster_name", maintain_order=True
    ):
        store_ids = stores["store_id"].cast(pl.String).to_list()
        model = fit_logit(
            train.filter(pl.col("store_id").is_in(store_ids)),
            sizes,
            fourier_terms=mcfg.demand.fourier_terms,
        )
        store_mean = float(np.mean(model.store_effect))
        delta = dict(zip(model.products, model.delta, strict=True))
        params.append(
            ClusterParams(
                cluster=int(cluster),
                name=name,
                alpha=model.alpha,
                alpha_cut=model.alpha_cut,
                beta_feature=model.beta_feature,
                beta_display=model.beta_display,
                # A product never sold in this cluster's stores gets a utility no one picks.
                delta=[delta.get(p, -50.0) + store_mean for p in ids],
                market_size=float(
                    sizes.filter(pl.col("store_id").is_in(store_ids))["market_size"].sum()
                ),
                stores=len(store_ids),
            )
        )
    c = mcfg.choice
    return Simulator(products, params, c.alpha_sd_ratio, c.shoppers_per_cluster, c.seed)


def implied_elasticities(
    train: pl.DataFrame, sizes: pl.DataFrame, plain: LogitModel, nested: LogitModel
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Elasticities the logit formulas imply at mean training prices and shares."""
    stats = (
        train.join(sizes, on="store_id")
        .group_by("product_id")
        .agg(
            price=pl.col("base_price").mean(),
            share=(pl.col("units") / pl.col("market_size")).mean(),
        )
    )
    order = pl.DataFrame({"product_id": plain.products})
    stats = order.join(stats, on="product_id", how="left")
    prices, shares = stats["price"].to_numpy(), stats["share"].to_numpy()
    e_plain = plain.implied_elasticities(prices, shares)
    e_nested = nested.implied_elasticities(prices, shares)
    own = stats.with_columns(
        plain_logit_own=pl.Series(np.diag(e_plain)), nested_logit_own=pl.Series(np.diag(e_nested))
    )
    pairs = [
        {
            "product_id": j,
            "other_product_id": k,
            "plain_logit": e_plain[a, b],
            "nested_logit": e_nested[a, b],
            "same_nest": nested.nests[a] == nested.nests[b],
        }
        for a, j in enumerate(plain.products)
        for b, k in enumerate(plain.products)
        if a != b
    ]
    return own, pl.DataFrame(pairs)


def run(cfg: ClientConfig, mcfg: ModelingConfig) -> None:
    panel = load_panel(cfg.client_id, mcfg)
    fold = final_fold(panel, mcfg.split)
    train = fold.train(panel)
    out = output_dir(cfg.client_id)

    plain, nested, sizes = fit_models(train, mcfg)
    log.info(
        "plain logit alpha %.3f, nested alpha %.3f sigma %.3f",
        plain.alpha,
        nested.alpha,
        nested.sigma,
    )
    (out / "choice_models.json").write_text(
        json.dumps(
            {
                "plain": plain.to_dict(),
                "nested": nested.to_dict(),
                "train_start": str(fold.train_start),
                "train_end": str(fold.train_end),
            },
            indent=1,
        )
    )

    own, pairs = implied_elasticities(train, sizes, plain, nested)
    regression = read_output(cfg.client_id, "elasticities.csv").with_columns(
        pl.col("product_id").cast(pl.String)
    )
    regression = regression.filter(
        pl.col("model").is_in(["M2 cross-price", "M3 hierarchical"])
    ).pivot(on="model", index="product_id", values="own_elasticity")
    own.join(regression, on="product_id", how="left").write_csv(out / "logit_elasticities.csv")
    pairs.write_csv(out / "logit_cross_elasticities.csv")

    products = current_products(read_table(cfg.client_id, "sales"), cfg.client_id)
    store_clusters = read_output(cfg.client_id, "store_clusters.csv")
    sim = build_simulator(train, sizes, products, store_clusters, mcfg)
    (out / "simulator.json").write_text(json.dumps(sim.to_dict(), default=str))

    promo = pl.concat([plan(sim, p, mcfg.promo_scenarios) for p in sim.ids])
    promo.join(products.select("product_id", "name", "size_oz"), on="product_id").write_csv(
        out / "promo_plan.csv"
    )

    o = mcfg.optimizer
    rules = build_rules(
        products, o.max_price_change, o.price_ending_cents, o.max_category_volume_drop
    )
    wape = _backtest_wape(cfg.client_id)
    summary = {}
    for objective in ("revenue", "margin"):
        result = optimize(sim, rules, objective, o.max_passes)
        log.info(
            "optimizer %s: %s via %s, gain %.0f",
            objective,
            result.status,
            result.method,
            result.gain,
        )
        summary[objective] = {
            "status": result.status,
            "method": result.method,
            "current_weekly_value": result.current_value,
            "optimized_weekly_value": result.value,
            "weekly_gain": result.gain,
            "gain_pct": result.gain / result.current_value if result.current_value else None,
            "current_weekly_units": result.current_units,
            "optimized_weekly_units": result.units,
            "wape": wape,
            "error_band_weekly": None if wape is None else wape * result.value,
            "binding": result.binding,
            "broken_today": result.broken_today,
            "prices": [
                {
                    "product_id": p,
                    "name": n,
                    "size_oz": s,
                    "current": float(c),
                    "recommended": float(r),
                }
                for p, n, s, c, r in zip(
                    sim.ids,
                    products["name"],
                    products["size_oz"],
                    result.current_prices,
                    result.prices,
                    strict=True,
                )
            ],
        }
    summary["default_objective"] = o.objective
    (out / "optimizer.json").write_text(json.dumps(summary, indent=1))


def _backtest_wape(client: str) -> float | None:
    """The nested logit's held-out WAPE on chain-wide weekly category units, the error band
    the optimizer gain is reported with."""
    try:
        table = read_output(client, "backtest_summary.csv")
    except FileNotFoundError:
        return None
    row = table.filter((pl.col("model") == "Nested logit") & (pl.col("segment") == "all"))
    return (
        float(row["wape_chain_week"][0])
        if row.height and "wape_chain_week" in row.columns
        else None
    )
