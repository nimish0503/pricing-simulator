"""Forecast and pricing-quality metrics. All are ratios of sums, so weeks with zero actual
sales don't blow anything up the way they do in MAPE."""

import numpy as np
import polars as pl


def wape(actual: np.ndarray, forecast: np.ndarray) -> float:
    """sum |actual - forecast| / sum actual."""
    return float(np.abs(actual - forecast).sum() / actual.sum())


def bias(actual: np.ndarray, forecast: np.ndarray) -> float:
    """sum (forecast - actual) / sum actual. Positive means over-forecasting."""
    return float((forecast - actual).sum() / actual.sum())


def mase(actual: np.ndarray, forecast: np.ndarray, naive: np.ndarray) -> float:
    """Absolute error relative to M0 on the same rows. Above 1 is worse than naive."""
    return float(np.abs(actual - forecast).sum() / np.abs(actual - naive).sum())


def promo_lift_calibration(actual: np.ndarray, forecast: np.ndarray, naive: np.ndarray) -> float:
    """Predicted incremental units over actual incremental units in promo weeks, both measured
    against the naive no-promo baseline. 1 is calibrated, below 1 under-predicts the lift."""
    actual_lift = (actual - naive).sum()
    return float((forecast - naive).sum() / actual_lift) if actual_lift > 0 else float("nan")


def direction_accuracy(frame: pl.DataFrame, min_change: float) -> float:
    """Share of held-out week-on-week price changes (|change| >= min_change, consecutive weeks,
    same store-product) where the forecast moved units in the same direction as the actuals."""
    changes = (
        frame.sort("store_id", "product_id", "week_start")
        .with_columns(
            gap=pl.col("week_start").diff().dt.total_days().over("store_id", "product_id"),
            d_price=(pl.col("unit_price") / pl.col("unit_price").shift() - 1).over(
                "store_id", "product_id"
            ),
            d_actual=pl.col("actual").diff().over("store_id", "product_id"),
            d_forecast=pl.col("forecast").diff().over("store_id", "product_id"),
        )
        .filter(
            (pl.col("gap") == 7)
            & (pl.col("d_price").abs() >= min_change)
            & (pl.col("d_actual") != 0)
        )
    )
    if changes.is_empty():
        return float("nan")
    return float((changes["d_actual"].sign() == changes["d_forecast"].sign()).mean())
