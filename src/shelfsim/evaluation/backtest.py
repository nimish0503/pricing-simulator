"""Rolling-origin backtest of the demand ladder and the nested logit on the same folds."""

import logging

import numpy as np
import polars as pl

from shelfsim.choice import fit_models
from shelfsim.config import ModelingConfig
from shelfsim.demand import ladder, plausibility
from shelfsim.evaluation import metrics
from shelfsim.splits import rolling_folds

log = logging.getLogger(__name__)

NESTED = "Nested logit"
NAIVE = "M0 naive"


def run_folds(panel: pl.DataFrame, mcfg: ModelingConfig) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Predictions for every held-out row and elasticities per model and fold."""
    predictions, elasticities = [], []
    for fold in rolling_folds(panel, mcfg.split, mcfg.backtest):
        train, test = fold.train(panel), fold.test(panel)
        log.info(
            "fold %d: train %s to %s, test to %s (%d rows)",
            fold.number,
            fold.train_start,
            fold.train_end,
            fold.test_end,
            test.height,
        )
        keep = test.select(
            "store_id",
            "product_id",
            "week_start",
            "unit_price",
            "on_price_cut",
            actual=pl.col("units").cast(pl.Float64),
        )
        for model in ladder(mcfg):
            model.fit(train)
            predictions.append(
                keep.with_columns(
                    forecast=pl.Series(model.predict(test)),
                    model=pl.lit(model.name),
                    fold=pl.lit(fold.number),
                )
            )
            elasticities.append(
                model.elasticities()
                .select("product_id", "own_elasticity")
                .with_columns(model=pl.lit(model.name), fold=pl.lit(fold.number))
            )
        _, nested, sizes = fit_models(train, mcfg)
        predictions.append(
            keep.with_columns(
                forecast=pl.Series(nested.predict_units(test, sizes)),
                model=pl.lit(NESTED),
                fold=pl.lit(fold.number),
            )
        )
        elasticities.append(
            _nested_own(train, sizes, nested).with_columns(
                model=pl.lit(NESTED), fold=pl.lit(fold.number)
            )
        )
    return pl.concat(predictions), pl.concat(elasticities)


def _nested_own(train: pl.DataFrame, sizes: pl.DataFrame, model) -> pl.DataFrame:
    stats = pl.DataFrame({"product_id": model.products}).join(
        train.join(sizes, on="store_id")
        .group_by("product_id")
        .agg(
            price=pl.col("base_price").mean(),
            share=(pl.col("units") / pl.col("market_size")).mean(),
        ),
        on="product_id",
        how="left",
    )
    e = model.implied_elasticities(stats["price"].to_numpy(), stats["share"].to_numpy())
    return pl.DataFrame({"product_id": model.products, "own_elasticity": np.diag(e)})


def summarize(
    predictions: pl.DataFrame, elasticities: pl.DataFrame, mcfg: ModelingConfig
) -> tuple[pl.DataFrame, pl.DataFrame]:
    naive = predictions.filter(pl.col("model") == NAIVE).select(
        "store_id", "product_id", "week_start", "fold", naive=pl.col("forecast")
    )
    rows = predictions.join(naive, on=["store_id", "product_id", "week_start", "fold"])
    segments = {
        "all": pl.lit(True),
        "promo": pl.col("on_price_cut"),
        "non-promo": ~pl.col("on_price_cut"),
    }

    def score(frame: pl.DataFrame) -> dict:
        a, f, n = (frame[c].to_numpy() for c in ("actual", "forecast", "naive"))
        return {
            "rows": frame.height,
            "actual_units": float(a.sum()),
            "wape": metrics.wape(a, f),
            "bias": metrics.bias(a, f),
            "mase": metrics.mase(a, f, n),
        }

    per_fold, overall = [], []
    for (model,), by_model in rows.group_by("model", maintain_order=True):
        for name, condition in segments.items():
            segment = by_model.filter(condition)
            if segment.is_empty():
                continue
            for (fold,), by_fold in segment.group_by("fold", maintain_order=True):
                per_fold.append({"model": model, "segment": name, "fold": fold, **score(by_fold)})
            summary = {"model": model, "segment": name, **score(segment)}
            if name == "all":
                # Error on chain-wide weekly category units: what an optimizer gain on the
                # category total is up against. Store-level errors partly cancel here.
                chain = by_model.group_by("fold", "week_start").agg(
                    pl.col("actual").sum(), pl.col("forecast").sum()
                )
                summary["wape_chain_week"] = metrics.wape(
                    chain["actual"].to_numpy(), chain["forecast"].to_numpy()
                )
                summary["direction_accuracy"] = float(
                    np.nanmean(
                        [
                            metrics.direction_accuracy(f, mcfg.backtest.price_change_min)
                            for _, f in by_model.group_by("fold")
                        ]
                    )
                )
            if name == "promo":
                summary["promo_lift_calibration"] = metrics.promo_lift_calibration(
                    *(segment[c].to_numpy() for c in ("actual", "forecast", "naive"))
                )
            overall.append(summary)

    known = plausibility(elasticities.drop_nulls("own_elasticity"), mcfg)
    # Elasticity stability: sd of each product's elasticity across folds, median over products.
    stability = (
        known.group_by("model", "product_id")
        .agg(sd=pl.col("own_elasticity").std())
        .group_by("model")
        .agg(elasticity_sd=pl.col("sd").median())
        .join(known.group_by("model").agg(plausibility=pl.col("plausible").mean()), on="model")
    )
    summary = pl.DataFrame(overall).join(stability, on="model", how="left")
    return pl.DataFrame(per_fold), summary
