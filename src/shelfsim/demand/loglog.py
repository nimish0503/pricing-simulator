"""M1: log-log OLS per product with store fixed effects."""

import numpy as np
import polars as pl
import statsmodels.api as sm

from shelfsim.demand.base import StoreEffects, elasticity_frame, smearing
from shelfsim.demand.features import fit_rows, fourier_columns, varying

Z95 = 1.959964


def linear_predict(d: pl.DataFrame, coefs: pl.DataFrame, columns: list[str]) -> np.ndarray:
    """exp(store-product mean + sum of coef * demeaned regressor) * smearing factor."""
    out = d.join(coefs, on="product_id", how="left")
    linear = pl.col("mean_ln_units")
    for c in columns:
        if f"b_{c}" in out.columns:
            linear = linear + pl.col(f"b_{c}").fill_null(0.0) * pl.col(f"d_{c}")
    units = linear.exp() * pl.col("smear").fill_null(1.0)
    return out.select(units.fill_null(0.0)).to_series().to_numpy()


def promo_row(coefs: dict[str, float]) -> dict:
    """Promo response read off the coefficients.

    promo_elasticity is d ln units / d ln price for a small temporary cut: units move by
    exp(gamma * depth) and depth = 1 - p / base, so at depth 0 the elasticity is -gamma.
    """
    return {
        "promo_elasticity": -coefs["discount_depth"] if "discount_depth" in coefs else None,
        "feature_lift": float(np.expm1(coefs["feature"])) if "feature" in coefs else None,
        "display_lift": float(np.expm1(coefs["display"])) if "display" in coefs else None,
    }


class LogLog:
    """Per product j, on weeks with units > 0:

        ln(units_sjt) = a_sj + beta_j ln(base_price_sjt) + gamma_j depth_sjt
                        + phi_j feature_sjt + psi_j display_sjt + fourier(week of year) + e

    a_sj is a store fixed effect (within transform). beta_j is the regular-price elasticity;
    gamma_j is the response to a temporary cut, kept separate because shoppers react more to a
    cut than to a permanent change. Standard errors are clustered by week, since prices move
    chain-wide in a given week. A product with no regular-price change in training gets no
    elasticity.
    """

    name = "M1 log-log"

    def __init__(self, fourier_terms: int, exclude_outliers: bool):
        self.columns = [
            "ln_base_price",
            "discount_depth",
            "feature",
            "display",
            *fourier_columns(fourier_terms),
        ]
        self.exclude_outliers = exclude_outliers

    def fit(self, train: pl.DataFrame) -> "LogLog":
        rows = fit_rows(train, self.exclude_outliers)
        self.effects = StoreEffects(self.columns).fit(rows)
        demeaned = self.effects.transform(rows)
        coef_rows, self.summary = [], []
        for (product_id,), group in demeaned.group_by("product_id"):
            names = [
                c
                for c in self.columns
                if f"d_{c}" in varying(group, [f"d_{c}" for c in self.columns])
            ]
            x = group.select(f"d_{c}" for c in names).to_numpy()
            y = group["d_ln_units"].to_numpy()
            weeks = group["week_start"].rank("dense").cast(pl.Int64).to_numpy()
            result = sm.OLS(y, x).fit(cov_type="cluster", cov_kwds={"groups": weeks})
            params = dict(zip(names, result.params, strict=True))
            se = dict(zip(names, result.bse, strict=True))
            coef_rows.append(
                {
                    "product_id": product_id,
                    **{f"b_{c}": v for c, v in params.items()},
                    "smear": smearing(result.resid),
                }
            )
            beta, beta_se = params.get("ln_base_price"), se.get("ln_base_price")
            self.summary.append(
                {
                    "product_id": product_id,
                    "own_elasticity": beta,
                    "own_lower": None if beta is None else beta - Z95 * beta_se,
                    "own_upper": None if beta is None else beta + Z95 * beta_se,
                    "cross_elasticity": None,
                    "cross_lower": None,
                    "cross_upper": None,
                    **promo_row(params),
                }
            )
        self.coefs = pl.DataFrame(coef_rows)
        return self

    def predict(self, frame: pl.DataFrame) -> np.ndarray:
        return linear_predict(self.effects.transform(frame), self.coefs, self.columns)

    def elasticities(self) -> pl.DataFrame:
        return elasticity_frame(self.summary)
