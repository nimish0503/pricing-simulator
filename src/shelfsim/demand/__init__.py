"""Fit the M0 to M4 ladder on the training period and write elasticities."""

import logging

import polars as pl

from shelfsim.config import ClientConfig, ModelingConfig
from shelfsim.demand.base import DemandModel
from shelfsim.demand.crossprice import CrossPrice
from shelfsim.demand.features import build_panel
from shelfsim.demand.gbm import GradientBoosting
from shelfsim.demand.hierarchical import Hierarchical
from shelfsim.demand.loglog import LogLog
from shelfsim.demand.naive import Naive
from shelfsim.io import output_dir, read_output, read_table
from shelfsim.splits import final_fold

log = logging.getLogger(__name__)


def ladder(mcfg: ModelingConfig) -> list[DemandModel]:
    d = mcfg.demand
    return [
        Naive(d.naive_window_weeks),
        LogLog(d.fourier_terms, d.exclude_outliers_from_fit),
        CrossPrice(d.fourier_terms, d.exclude_outliers_from_fit),
        Hierarchical(
            d.fourier_terms,
            d.exclude_outliers_from_fit,
            d.hb_max_rows_per_product,
            d.hb_draws,
            d.hb_tune,
            d.hb_chains,
            d.seed,
        ),
        GradientBoosting(
            d.fourier_terms,
            d.gbm_lag_weeks,
            d.gbm_rounds,
            d.gbm_learning_rate,
            d.gbm_num_leaves,
            d.elasticity_perturbation,
            d.seed,
        ),
    ]


def load_panel(client: str, mcfg: ModelingConfig) -> pl.DataFrame:
    clusters = read_output(client, "product_clusters.csv").with_columns(
        pl.col("product_id").cast(pl.String)
    )
    return build_panel(
        read_table(client, "sales"), clusters, mcfg.demand.fourier_terms, mcfg.demand.gbm_lag_weeks
    )


def plausibility(frame: pl.DataFrame, mcfg: ModelingConfig) -> pl.DataFrame:
    p = mcfg.plausibility
    return frame.with_columns(
        plausible=pl.col("own_elasticity").is_between(p.own_elasticity_min, p.own_elasticity_max)
    )


def run(cfg: ClientConfig, mcfg: ModelingConfig) -> None:
    panel = load_panel(cfg.client_id, mcfg)
    fold = final_fold(panel, mcfg.split)
    train = fold.train(panel)
    log.info("training on %s to %s, %d rows", fold.train_start, fold.train_end, train.height)

    products = read_table(cfg.client_id, "products").select("product_id", "name", "size_oz")
    tables, pairs = [], []
    for model in ladder(mcfg):
        model.fit(train)
        log.info("fitted %s", model.name)
        tables.append(model.elasticities().with_columns(model=pl.lit(model.name)))
        if hasattr(model, "pairs"):
            pairs.append(pl.DataFrame(model.pairs).with_columns(model=pl.lit(model.name)))

    out = output_dir(cfg.client_id)
    elasticities = plausibility(pl.concat(tables), mcfg).join(products, on="product_id", how="left")
    elasticities.with_columns(
        train_start=pl.lit(fold.train_start), train_end=pl.lit(fold.train_end)
    ).select(
        "model",
        "product_id",
        "name",
        "size_oz",
        *[c for c in elasticities.columns if c not in ("model", "product_id", "name", "size_oz")],
        "train_start",
        "train_end",
    ).write_csv(out / "elasticities.csv")
    if pairs:
        pl.concat(pairs).write_csv(out / "cross_elasticities.csv")
