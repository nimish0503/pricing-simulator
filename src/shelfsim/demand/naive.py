"""M0: the baseline every model has to beat."""

import numpy as np
import polars as pl

from shelfsim.demand.base import elasticity_frame
from shelfsim.demand.features import KEYS


class Naive:
    """units_hat = mean units over the store-product's last `window` regular-price weeks in
    training, flat across the whole forecast horizon. No price response, so no elasticities."""

    name = "M0 naive"

    def __init__(self, window: int):
        self.window = window

    def fit(self, train: pl.DataFrame) -> "Naive":
        regular = train.filter(~pl.col("on_price_cut") & ~pl.col("is_outlier"))
        self.level = (
            regular.sort("week_start")
            .group_by(KEYS)
            .agg(level=pl.col("units").tail(self.window).mean())
        )
        self.product_level = self.level.group_by("product_id").agg(
            product_level=pl.col("level").mean()
        )
        return self

    def predict(self, frame: pl.DataFrame) -> np.ndarray:
        out = frame.join(self.level, on=KEYS, how="left").join(
            self.product_level, on="product_id", how="left"
        )
        return (
            out.select(pl.coalesce("level", "product_level").fill_null(0.0)).to_series().to_numpy()
        )

    def elasticities(self) -> pl.DataFrame:
        return elasticity_frame([])
