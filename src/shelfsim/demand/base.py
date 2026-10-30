"""The interface every demand model implements, and the store fixed-effects transform the
log-log models share.

fit(train) -> model and predict(model, frame) -> units, so the backtest can treat M0 to M4
the same way.
"""

from typing import Protocol

import numpy as np
import polars as pl

from shelfsim.demand.features import KEYS

ELASTICITY_COLUMNS = {
    "product_id": pl.String,
    "own_elasticity": pl.Float64,
    "own_lower": pl.Float64,
    "own_upper": pl.Float64,
    "cross_elasticity": pl.Float64,
    "cross_lower": pl.Float64,
    "cross_upper": pl.Float64,
    "promo_elasticity": pl.Float64,
    "feature_lift": pl.Float64,
    "display_lift": pl.Float64,
}


class DemandModel(Protocol):
    name: str

    def fit(self, train: pl.DataFrame) -> "DemandModel": ...

    def predict(self, frame: pl.DataFrame) -> np.ndarray: ...

    def elasticities(self) -> pl.DataFrame: ...


def fit(model: DemandModel, train: pl.DataFrame) -> DemandModel:
    return model.fit(train)


def predict(model: DemandModel, frame: pl.DataFrame) -> np.ndarray:
    return model.predict(frame)


def elasticity_frame(rows: list[dict]) -> pl.DataFrame:
    return (
        pl.DataFrame(rows, schema=ELASTICITY_COLUMNS, orient="row")
        if rows
        else pl.DataFrame(schema=ELASTICITY_COLUMNS)
    )


class StoreEffects:
    """Within transform: subtract each store-product's training mean from y and every regressor.

    That is the same as a fixed effect per store-product, and it lets each product's regression
    run on a few thousand demeaned rows instead of a design matrix with 90 store dummies. At
    predict time the training means come back in; a store-product never seen in training uses
    its product's mean.
    """

    def __init__(self, columns: list[str]):
        self.columns = columns

    def fit(self, frame: pl.DataFrame) -> "StoreEffects":
        stats = [pl.col(c).mean().alias(f"mean_{c}") for c in ["ln_units", *self.columns]]
        self.by_store = frame.group_by(KEYS).agg(stats)
        self.by_product = frame.group_by("product_id").agg(stats)
        return self

    def transform(self, frame: pl.DataFrame) -> pl.DataFrame:
        means = [f"mean_{c}" for c in ["ln_units", *self.columns]]
        out = frame.join(self.by_store, on=KEYS, how="left").join(
            self.by_product.rename({m: f"{m}_p" for m in means}), on="product_id", how="left"
        )
        out = out.with_columns(pl.coalesce(m, f"{m}_p").alias(m) for m in means).drop(
            f"{m}_p" for m in means
        )
        return out.with_columns(
            (pl.col(c).fill_null(pl.col(f"mean_{c}")) - pl.col(f"mean_{c}")).alias(f"d_{c}")
            for c in self.columns
        ).with_columns(d_ln_units=pl.col("ln_units") - pl.col("mean_ln_units"))


def smearing(residuals: np.ndarray) -> float:
    """Duan's smearing factor: exp(E[ln y]) underestimates E[y], this corrects it."""
    return float(np.mean(np.exp(residuals))) if len(residuals) else 1.0
