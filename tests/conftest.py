"""Synthetic data with known parameters, so every model can be checked for recovering them."""

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from shelfsim.config import load_modeling
from shelfsim.demand.features import build_panel

TRUE_ELASTICITY = -2.0
TRUE_PROMO = 3.0  # gamma on discount depth
TRUE_FEATURE = 0.4


def synthetic_sales(
    n_stores=8, n_products=4, n_weeks=120, elasticity=TRUE_ELASTICITY, noise=0.15, seed=0
) -> pl.DataFrame:
    """Canonical sales rows where ln(units) = a_sj + elasticity * ln(base) + 3.0 * depth +
    0.4 * feature + seasonality + N(0, noise)."""
    rng = np.random.default_rng(seed)
    rows = []
    start = date(2010, 1, 4)
    for j in range(n_products):
        # Regular price moves in steps: a few permanent changes over the period.
        steps = np.cumsum(rng.choice([0.0, 0.0, 0.0, 0.0, 0.08, -0.08], size=n_weeks))
        base_path = 3.0 * np.exp(steps * 0.5)
        for s in range(n_stores):
            level = 3.5 + 0.3 * rng.standard_normal() + 0.2 * j
            store_base = base_path * (1 + 0.03 * rng.standard_normal())
            for t in range(n_weeks):
                promo = rng.random() < 0.15
                depth = rng.choice([0.1, 0.2, 0.3]) if promo else 0.0
                feature = promo and rng.random() < 0.5
                week = start + timedelta(weeks=t)
                season = 0.1 * np.sin(2 * np.pi * week.timetuple().tm_yday / 365.25)
                ln_units = (
                    level
                    + elasticity * np.log(store_base[t])
                    + TRUE_PROMO * depth
                    + TRUE_FEATURE * feature
                    + season
                    + noise * rng.standard_normal()
                )
                units = max(int(round(np.exp(ln_units))), 0)
                price = store_base[t] * (1 - depth)
                rows.append(
                    {
                        "client_id": "synthetic",
                        "store_id": str(s),
                        "product_id": f"p{j}",
                        "week_start": week,
                        "units": units,
                        "revenue": units * price,
                        "unit_price": price,
                        "base_price": store_base[t],
                        "discount_depth": depth,
                        "on_feature": feature,
                        "on_display": False,
                        "on_price_cut": promo,
                        "unit_cost": price * 0.7,
                        "row_status": "observed" if units > 0 else "zero_sales",
                        "is_outlier": False,
                    }
                )
    return pl.DataFrame(rows)


@pytest.fixture(scope="session")
def mcfg():
    return load_modeling()


@pytest.fixture(scope="session")
def synthetic_panel(mcfg) -> pl.DataFrame:
    sales = synthetic_sales()
    clusters = pl.DataFrame({"product_id": ["p0", "p1", "p2", "p3"], "cluster": [0, 0, 1, 1]})
    return build_panel(sales, clusters, mcfg.demand.fourier_terms, mcfg.demand.gbm_lag_weeks)
