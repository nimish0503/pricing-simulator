"""Logit share model estimated with the Berry inversion. numpy only.

Each store-week is a market of size M_s. A shopper buys one of the J products or nothing
(the outside option, utility 0). With s_j = units_j / M_s and s_0 = 1 - sum_j s_j, the plain
logit gives ln(s_j) - ln(s_0) = V_j, linear in the parameters, so OLS recovers them. The nested
logit adds sigma * ln(s_j|g), the share of j within its nest g (ADR 0006 has M, nested.py has
the share formulas).
"""

from dataclasses import asdict, dataclass, field

import numpy as np
import polars as pl

from shelfsim.choice.nested import nested_elasticities, nested_shares
from shelfsim.demand.features import fourier_columns


def market_sizes(train: pl.DataFrame, peak_multiple: float) -> pl.DataFrame:
    """M_s = peak_multiple x the store's highest weekly category units in training."""
    weekly = train.group_by("store_id", "week_start").agg(pl.col("units").sum())
    return weekly.group_by("store_id").agg(market_size=pl.col("units").max() * peak_multiple)


@dataclass
class LogitModel:
    products: list[str]
    stores: list[str]
    delta: list[float]  # product fixed effects
    store_effect: list[float]  # first store is the reference, 0
    alpha: float  # utility per $ of regular (base) price
    alpha_cut: float  # utility per $ of temporary discount
    beta_feature: float
    beta_display: float
    season: list[float]
    sigma: float = 0.0  # nesting parameter; 0 is the plain logit
    nests: list[int] = field(default_factory=list)
    std_errors: dict[str, float] = field(default_factory=dict)
    n_obs: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "LogitModel":
        return cls(**data)

    def utilities(self, frame: pl.DataFrame) -> np.ndarray:
        p = {pid: i for i, pid in enumerate(self.products)}
        s = {sid: i for i, sid in enumerate(self.stores)}
        j = frame["product_id"].replace_strict(p, default=-1).to_numpy()
        st = frame["store_id"].replace_strict(s, default=-1).to_numpy()
        delta = np.where(j >= 0, np.asarray(self.delta)[np.clip(j, 0, None)], np.nan)
        store = np.where(st >= 0, np.asarray(self.store_effect)[np.clip(st, 0, None)], 0.0)
        season = (
            frame.select(fourier_columns(len(self.season) // 2)).to_numpy()
            @ np.asarray(self.season)
            if self.season
            else 0.0
        )
        cut = (frame["base_price"] - frame["unit_price"]).clip(lower_bound=0).to_numpy()
        return (
            delta
            + store
            + season
            + self.alpha * frame["base_price"].to_numpy()
            + self.alpha_cut * cut
            + self.beta_feature * frame["feature"].to_numpy()
            + self.beta_display * frame["display"].to_numpy()
        )

    def predict_units(self, frame: pl.DataFrame, sizes: pl.DataFrame) -> np.ndarray:
        """Units per row: shares within each store-week market times the market size."""
        # A store with no sales in the training window gets the median store's market size
        # and no store effect.
        frame = (
            frame.with_row_index("_row")
            .join(sizes, on="store_id", how="left")
            .with_columns(pl.col("market_size").fill_null(sizes["market_size"].median()))
        )
        v = self.utilities(frame)
        nest_of = dict(zip(self.products, self.nests or [0] * len(self.products), strict=True))
        frame = frame.with_columns(
            _v=pl.Series(v), _nest=pl.col("product_id").replace_strict(nest_of, default=-1)
        )
        out = np.zeros(frame.height)
        for _, market in frame.filter(pl.col("_v").is_not_nan()).group_by("store_id", "week_start"):
            shares = nested_shares(market["_v"].to_numpy(), market["_nest"].to_numpy(), self.sigma)
            out[market["_row"].to_numpy()] = shares * market["market_size"].to_numpy()
        return out

    def implied_elasticities(self, prices: np.ndarray, shares: np.ndarray) -> np.ndarray:
        """E[j, k] = d ln s_j / d ln p_k at the given prices and shares (rows j, columns k)."""
        return nested_elasticities(
            self.alpha, prices, shares, np.asarray(self.nests or [0] * len(prices)), self.sigma
        )


def fit_logit(
    train: pl.DataFrame,
    sizes: pl.DataFrame,
    nests: dict[str, int] | None = None,
    fourier_terms: int = 0,
) -> LogitModel:
    """OLS on ln(s_j) - ln(s_0) with product and store dummies.

    With nests, the regressor ln(s_j|g) is correlated with the error (a demand shock to j
    raises its within-nest share), so OLS sigma is biased upward. It's the standard first
    pass; an instrument for it would be the next step.
    """
    frame = train.filter(pl.col("units") > 0).join(sizes, on="store_id")
    market = ["store_id", "week_start"]
    frame = frame.with_columns(
        s=pl.col("units") / pl.col("market_size"),
        s_inside=pl.col("units").sum().over(market) / pl.col("market_size"),
    ).filter(pl.col("s_inside") < 1)
    frame = frame.with_columns(
        y=(pl.col("s").log() - (1 - pl.col("s_inside")).log()),
        cut=(pl.col("base_price") - pl.col("unit_price")).clip(lower_bound=0),
    )
    if nests:
        frame = frame.with_columns(nest=pl.col("product_id").replace_strict(nests))
        frame = frame.with_columns(
            ln_within=(pl.col("units") / pl.col("units").sum().over(*market, "nest")).log()
        )

    products = sorted(frame["product_id"].unique())
    stores = sorted(frame["store_id"].unique())
    season = fourier_columns(fourier_terms)
    named = ["base_price", "cut", "feature", "display", *season, *(["ln_within"] if nests else [])]
    named = [c for c in named if frame[c].n_unique() > 1]
    j = frame["product_id"].replace_strict({p: i for i, p in enumerate(products)}).to_numpy()
    st = frame["store_id"].replace_strict({s: i for i, s in enumerate(stores)}).to_numpy()
    n = frame.height
    x = np.zeros((n, len(products) + len(stores) - 1 + len(named)))
    x[np.arange(n), j] = 1.0
    has_store = st > 0
    x[np.arange(n)[has_store], len(products) + st[has_store] - 1] = 1.0
    x[:, len(products) + len(stores) - 1 :] = frame.select(named).to_numpy()
    y = frame["y"].to_numpy()

    coef, *_ = np.linalg.lstsq(x, y, rcond=None)
    resid = y - x @ coef
    dof = max(n - x.shape[1], 1)
    xtx_inv = np.linalg.pinv(x.T @ x)
    se = np.sqrt(np.clip(np.diag(xtx_inv) * (resid @ resid) / dof, 0, None))
    offset = len(products) + len(stores) - 1
    params = dict(zip(named, coef[offset:], strict=True))
    errors = dict(zip(named, se[offset:], strict=True))
    return LogitModel(
        products=products,
        stores=stores,
        delta=coef[: len(products)].tolist(),
        store_effect=[0.0, *coef[len(products) : offset].tolist()],
        alpha=float(params["base_price"]),
        alpha_cut=float(params.get("cut", params["base_price"])),
        beta_feature=float(params.get("feature", 0.0)),
        beta_display=float(params.get("display", 0.0)),
        season=[float(params.get(c, 0.0)) for c in season],
        sigma=float(np.clip(params.get("ln_within", 0.0), 0.0, 0.95)),
        nests=[nests[p] for p in products] if nests else [],
        std_errors={k: float(v) for k, v in errors.items()},
        n_obs=n,
    )
