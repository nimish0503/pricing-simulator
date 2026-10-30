"""M2: log-log pooled within product cluster, with a cross-price term."""

import numpy as np
import polars as pl
import statsmodels.api as sm

from shelfsim.demand.base import StoreEffects, elasticity_frame, smearing
from shelfsim.demand.features import fit_rows, fourier_columns, varying
from shelfsim.demand.loglog import Z95, linear_predict, promo_row


class CrossPrice:
    """Per product cluster c, pooled over its products j:

        ln(units_sjt) = a_sj + beta_j ln(base_price_sjt) + delta_j ln(pbar_-j,st)
                        + gamma_c depth + phi_c feature + psi_c display + fourier + e

    pbar_-j is the mean unit price of the other products in the cluster in that store and week.
    delta_j is product j's cross-price elasticity against the rest of its cluster: positive
    means j gains when they get dearer, which is cannibalization seen from the other side.
    Own and cross slopes are per product; promo and seasonal effects are shared by the cluster,
    which is what pooling buys. Standard errors are clustered by week.
    """

    name = "M2 cross-price"

    def __init__(self, fourier_terms: int, exclude_outliers: bool):
        self.shared = ["discount_depth", "feature", "display", *fourier_columns(fourier_terms)]
        self.columns = ["ln_base_price", "ln_cross_price", *self.shared]
        self.exclude_outliers = exclude_outliers

    def fit(self, train: pl.DataFrame) -> "CrossPrice":
        rows = fit_rows(train, self.exclude_outliers)
        self.effects = StoreEffects(self.columns).fit(rows)
        demeaned = self.effects.transform(rows)
        coef_rows, self.summary, self.pairs = [], [], []
        for (cluster,), group in demeaned.group_by("cluster"):
            products = sorted(group["product_id"].unique())
            shared = [
                c
                for c in self.shared
                if f"d_{c}" in varying(group, [f"d_{c}" for c in self.shared])
            ]
            blocks, names = [], []
            is_product = {
                p: (group["product_id"] == p).cast(pl.Float64).to_numpy() for p in products
            }
            for p in products:
                for c in ("ln_base_price", "ln_cross_price"):
                    column = group[f"d_{c}"].to_numpy() * is_product[p]
                    if np.any(column != 0):
                        blocks.append(column)
                        names.append((p, c))
            for c in shared:
                blocks.append(group[f"d_{c}"].to_numpy())
                names.append((None, c))
            x = np.column_stack(blocks)
            y = group["d_ln_units"].to_numpy()
            weeks = group["week_start"].rank("dense").cast(pl.Int64).to_numpy()
            result = sm.OLS(y, x).fit(cov_type="cluster", cov_kwds={"groups": weeks})
            resid = y - x @ result.params
            product_codes = group["product_id"].to_numpy()
            shared_params = {
                c: v for (p, c), v in zip(names, result.params, strict=True) if p is None
            }
            for p in products:
                own = {
                    c: (v, s)
                    for (q, c), v, s in zip(names, result.params, result.bse, strict=True)
                    if q == p
                }
                params = {c: v for c, (v, _) in own.items()} | shared_params
                coef_rows.append(
                    {
                        "product_id": p,
                        **{f"b_{c}": v for c, v in params.items()},
                        "smear": smearing(resid[product_codes == p]),
                    }
                )
                beta, beta_se = own.get("ln_base_price", (None, None))
                cross, cross_se = own.get("ln_cross_price", (None, None))
                self.summary.append(
                    {
                        "product_id": p,
                        "own_elasticity": beta,
                        "own_lower": None if beta is None else beta - Z95 * beta_se,
                        "own_upper": None if beta is None else beta + Z95 * beta_se,
                        "cross_elasticity": cross,
                        "cross_lower": None if cross is None else cross - Z95 * cross_se,
                        "cross_upper": None if cross is None else cross + Z95 * cross_se,
                        **promo_row(params),
                    }
                )
                # A 1% rise in one of the n - 1 other products' prices moves their mean by
                # about 1 / (n - 1) percent, so the pairwise cross elasticity is delta / (n - 1).
                if cross is not None and len(products) > 1:
                    for q in products:
                        if q != p:
                            self.pairs.append(
                                {
                                    "product_id": p,
                                    "other_product_id": q,
                                    "cross_elasticity": cross / (len(products) - 1),
                                    "cluster": cluster,
                                }
                            )
        self.coefs = pl.DataFrame(coef_rows)
        return self

    def predict(self, frame: pl.DataFrame) -> np.ndarray:
        return linear_predict(self.effects.transform(frame), self.coefs, self.columns)

    def elasticities(self) -> pl.DataFrame:
        return elasticity_frame(self.summary)
