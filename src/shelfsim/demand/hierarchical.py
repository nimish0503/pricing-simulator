"""M3: hierarchical Bayes log-log, product elasticities partially pooled by cluster."""

import numpy as np
import polars as pl
import pymc as pm

from shelfsim.demand.base import StoreEffects, elasticity_frame
from shelfsim.demand.features import fit_rows, fourier_columns, varying
from shelfsim.demand.loglog import linear_predict


class Hierarchical:
    """Same regression as M1, on the same store-demeaned data, with

        beta_j  ~ Normal(mu_c[j], tau)       regular-price elasticity of product j in cluster c
        gamma_j ~ Normal(m_c[j], tau_g)      response to temporary discount depth
        mu_c    ~ Normal(-1, 2),  m_c ~ Normal(2, 3),  tau, tau_g ~ HalfNormal(1), HalfNormal(2)
        feature, display: one effect per cluster ~ Normal(0, 1)
        seasonality: shared Fourier terms ~ Normal(0, 1)
        y ~ Normal(mu, sigma_j),  sigma_j ~ HalfNormal(1)

    A product with little price variation gets pulled toward its cluster's mean instead of
    producing a wild estimate. The prior on mu_c is wide and not centred on the -2 used in the
    synthetic tests. Non-centred parameterisation for the product effects; at most
    max_rows_per_product rows per product (a fixed-seed sample) to keep NUTS to minutes.
    Intervals are 95% central credible intervals.
    """

    name = "M3 hierarchical"

    def __init__(
        self,
        fourier_terms: int,
        exclude_outliers: bool,
        max_rows_per_product: int,
        draws: int,
        tune: int,
        chains: int,
        seed: int,
    ):
        self.season = fourier_columns(fourier_terms)
        self.columns = ["ln_base_price", "discount_depth", "feature", "display", *self.season]
        self.exclude_outliers = exclude_outliers
        self.max_rows = max_rows_per_product
        self.draws, self.tune, self.chains, self.seed = draws, tune, chains, seed

    def fit(self, train: pl.DataFrame) -> "Hierarchical":
        rows = fit_rows(train, self.exclude_outliers)
        self.effects = StoreEffects(self.columns).fit(rows)
        demeaned = self.effects.transform(rows)
        sample = (
            demeaned.with_columns(
                r=pl.int_range(pl.len()).shuffle(seed=self.seed).over("product_id")
            )
            .filter(pl.col("r") < self.max_rows)
            .sort("product_id")
        )
        products = sorted(sample["product_id"].unique())
        clusters_of = dict(sample.select("product_id", "cluster").unique().iter_rows())
        cluster_ids = sorted(set(clusters_of.values()))
        product_cluster = np.array([cluster_ids.index(clusters_of[p]) for p in products])
        j = sample["product_id"].replace_strict({p: i for i, p in enumerate(products)}).to_numpy()
        c = product_cluster[j]
        promo_signals = [
            s for s in ("feature", "display") if f"d_{s}" in varying(sample, [f"d_{s}"])
        ]

        def col(name):
            return sample[f"d_{name}"].to_numpy()

        with pm.Model():
            mu = pm.Normal("mu", -1.0, 2.0, shape=len(cluster_ids))
            tau = pm.HalfNormal("tau", 1.0)
            z = pm.Normal("z", 0.0, 1.0, shape=len(products))
            beta = pm.Deterministic("beta", mu[product_cluster] + tau * z)
            m = pm.Normal("m", 2.0, 3.0, shape=len(cluster_ids))
            tau_g = pm.HalfNormal("tau_g", 2.0)
            zg = pm.Normal("zg", 0.0, 1.0, shape=len(products))
            gamma = pm.Deterministic("gamma", m[product_cluster] + tau_g * zg)
            season = pm.Normal("season", 0.0, 1.0, shape=len(self.season))
            mean = beta[j] * col("ln_base_price") + gamma[j] * col("discount_depth")
            mean = mean + pm.math.dot(
                sample.select(f"d_{s}" for s in self.season).to_numpy(), season
            )
            for signal in promo_signals:
                effect = pm.Normal(signal, 0.0, 1.0, shape=len(cluster_ids))
                mean = mean + effect[c] * col(signal)
            sigma = pm.HalfNormal("sigma", 1.0, shape=len(products))
            pm.Normal("y", mean, sigma[j], observed=sample["d_ln_units"].to_numpy())
            trace = pm.sample(
                draws=self.draws,
                tune=self.tune,
                chains=self.chains,
                cores=1,
                random_seed=self.seed,
                progressbar=False,
                target_accept=0.9,
            )

        post = trace.posterior
        beta_draws = post["beta"].values.reshape(-1, len(products))
        gamma_draws = post["gamma"].values.reshape(-1, len(products))
        season_mean = post["season"].values.reshape(-1, len(self.season)).mean(axis=0)
        signal_means = {
            s: post[s].values.reshape(-1, len(cluster_ids)).mean(axis=0) for s in promo_signals
        }
        self.divergences = int(trace.sample_stats["diverging"].values.sum())

        coef_rows, self.summary = [], []
        for i, p in enumerate(products):
            ci = product_cluster[i]
            params = {
                "ln_base_price": beta_draws[:, i].mean(),
                "discount_depth": gamma_draws[:, i].mean(),
            }
            params |= dict(zip(self.season, season_mean, strict=True))
            params |= {s: v[ci] for s, v in signal_means.items()}
            coef_rows.append({"product_id": p, **{f"b_{k}": v for k, v in params.items()}})
            self.summary.append(
                {
                    "product_id": p,
                    "own_elasticity": params["ln_base_price"],
                    "own_lower": float(np.quantile(beta_draws[:, i], 0.025)),
                    "own_upper": float(np.quantile(beta_draws[:, i], 0.975)),
                    "cross_elasticity": None,
                    "cross_lower": None,
                    "cross_upper": None,
                    "promo_elasticity": -params["discount_depth"],
                    "feature_lift": float(np.expm1(params["feature"]))
                    if "feature" in params
                    else None,
                    "display_lift": float(np.expm1(params["display"]))
                    if "display" in params
                    else None,
                }
            )
        self.coefs = pl.DataFrame(coef_rows)
        fitted = demeaned.join(self.coefs, on="product_id")
        linear = sum(pl.col(f"b_{k}") * pl.col(f"d_{k}") for k in params)
        resid = fitted.select((pl.col("d_ln_units") - linear).alias("r"), "product_id")
        # Duan smearing per product, as in M1.
        smear = resid.group_by("product_id").agg(smear=pl.col("r").exp().mean())
        self.coefs = self.coefs.join(smear, on="product_id")
        return self

    def predict(self, frame: pl.DataFrame) -> np.ndarray:
        return linear_predict(self.effects.transform(frame), self.coefs, self.columns)

    def elasticities(self) -> pl.DataFrame:
        return elasticity_frame(self.summary)
