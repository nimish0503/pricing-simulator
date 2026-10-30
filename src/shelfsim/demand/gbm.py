"""M4: LightGBM benchmark. Can ML forecast better than the econometric models?"""

import lightgbm as lgb
import numpy as np
import polars as pl

from shelfsim.demand.base import elasticity_frame
from shelfsim.demand.features import KEYS, fourier_columns


class GradientBoosting:
    """log1p(units) on price, promo, seasonality, store and product codes, the store-product's
    training mean, and lagged units at lags >= the forecast horizon (so a 13-week forecast
    never uses actuals from inside the horizon).

    Nothing ties the price response to a coefficient, so elasticity is measured: raise base and
    unit price by `perturbation` on the training rows of regular-price weeks, and take the mean
    change in log predicted units over ln(1 + perturbation). There is no standard error for
    that number, so the interval columns stay empty; the backtest reports how much it moves
    across folds instead.
    """

    name = "M4 LightGBM"

    def __init__(
        self,
        fourier_terms: int,
        lags: list[int],
        rounds: int,
        learning_rate: float,
        num_leaves: int,
        perturbation: float,
        seed: int,
    ):
        self.features = [
            "store_code",
            "product_code",
            "cluster",
            "ln_base_price",
            "discount_depth",
            "feature",
            "display",
            "ln_cross_price",
            "level",
            *fourier_columns(fourier_terms),
            *(f"lag{lag}" for lag in lags),
        ]
        self.params = {
            "objective": "regression",
            "learning_rate": learning_rate,
            "num_leaves": num_leaves,
            "min_data_in_leaf": 50,
            "feature_fraction": 0.9,
            "seed": seed,
            "verbosity": -1,
        }
        self.rounds = rounds
        self.perturbation = perturbation

    def _design(self, frame: pl.DataFrame) -> pl.DataFrame:
        out = frame.join(self.levels, on=KEYS, how="left").with_columns(
            store_code=pl.col("store_id").replace_strict(self.store_codes, default=-1),
            product_code=pl.col("product_id").replace_strict(self.product_codes, default=-1),
        )
        return out.select(self.features).cast(pl.Float64)

    def fit(self, train: pl.DataFrame) -> "GradientBoosting":
        rows = train.filter(~pl.col("is_outlier"))
        self.store_codes = {s: i for i, s in enumerate(sorted(rows["store_id"].unique()))}
        self.product_codes = {p: i for i, p in enumerate(sorted(rows["product_id"].unique()))}
        self.levels = rows.group_by(KEYS).agg(level=pl.col("units").cast(pl.Float64).log1p().mean())
        x = self._design(rows).to_numpy()
        y = rows["units"].cast(pl.Float64).log1p().to_numpy()
        data = lgb.Dataset(
            x,
            y,
            feature_name=self.features,
            categorical_feature=["store_code", "product_code", "cluster"],
            free_raw_data=False,
        )
        self.booster = lgb.train(self.params, data, num_boost_round=self.rounds)
        # log1p target predicts something close to a median; rescale to the training mean.
        fitted = np.expm1(self.booster.predict(x))
        self.scale = float(rows["units"].sum() / fitted.sum())
        self._train = rows
        return self

    def predict(self, frame: pl.DataFrame) -> np.ndarray:
        return (
            np.clip(np.expm1(self.booster.predict(self._design(frame).to_numpy())), 0, None)
            * self.scale
        )

    def elasticities(self) -> pl.DataFrame:
        base = self._train.filter(~pl.col("on_price_cut"))
        bumped = base.with_columns(
            ln_base_price=pl.col("ln_base_price") + np.log1p(self.perturbation),
        )
        before = np.log(self.predict(base) + 1e-9)
        after = np.log(self.predict(bumped) + 1e-9)
        change = pl.DataFrame(
            {"product_id": base["product_id"], "d": (after - before) / np.log1p(self.perturbation)}
        )
        own = change.group_by("product_id").agg(own_elasticity=pl.col("d").mean())
        rows = [
            {
                "product_id": r["product_id"],
                "own_elasticity": r["own_elasticity"],
                "own_lower": None,
                "own_upper": None,
                "cross_elasticity": None,
                "cross_lower": None,
                "cross_upper": None,
                "promo_elasticity": None,
                "feature_lift": None,
                "display_lift": None,
            }
            for r in own.iter_rows(named=True)
        ]
        return elasticity_frame(rows)
