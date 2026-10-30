import numpy as np
import polars as pl
import pytest

from shelfsim.demand.base import fit, predict
from shelfsim.demand.crossprice import CrossPrice
from shelfsim.demand.gbm import GradientBoosting
from shelfsim.demand.hierarchical import Hierarchical
from shelfsim.demand.loglog import LogLog
from shelfsim.demand.naive import Naive
from tests.conftest import TRUE_ELASTICITY, TRUE_PROMO

TOLERANCE = 0.25


def own(model) -> np.ndarray:
    return model.elasticities()["own_elasticity"].to_numpy()


def test_loglog_recovers_elasticity_and_promo(synthetic_panel):
    model = fit(LogLog(3, True), synthetic_panel)
    assert np.abs(own(model) - TRUE_ELASTICITY).max() < TOLERANCE
    promo = model.elasticities()["promo_elasticity"].to_numpy()
    assert np.abs(promo + TRUE_PROMO).max() < 0.4
    intervals = model.elasticities()
    assert (intervals["own_lower"] < TRUE_ELASTICITY).all()
    assert (intervals["own_upper"] > TRUE_ELASTICITY).all()


def test_crossprice_recovers_own_and_zero_cross(synthetic_panel):
    model = fit(CrossPrice(3, True), synthetic_panel)
    e = model.elasticities()
    assert np.abs(e["own_elasticity"].to_numpy() - TRUE_ELASTICITY).max() < TOLERANCE
    # The synthetic data has no cross-price effect, so the interval should cover 0.
    assert ((e["cross_lower"] < 0) & (e["cross_upper"] > 0)).mean() >= 0.75


def test_hierarchical_recovers_elasticity(synthetic_panel):
    model = fit(
        Hierarchical(3, True, max_rows_per_product=400, draws=300, tune=300, chains=1, seed=1),
        synthetic_panel,
    )
    assert np.abs(own(model) - TRUE_ELASTICITY).max() < TOLERANCE
    e = model.elasticities()
    assert (e["own_lower"] < e["own_elasticity"]).all() and (
        e["own_upper"] > e["own_elasticity"]
    ).all()


def test_naive_predicts_recent_regular_level(synthetic_panel):
    model = fit(Naive(4), synthetic_panel)
    forecast = predict(model, synthetic_panel)
    assert forecast.shape == (synthetic_panel.height,)
    assert (forecast > 0).all()
    assert model.elasticities().is_empty()


def test_gbm_fits_and_measures_price_response(synthetic_panel):
    model = fit(
        GradientBoosting(
            3, [13, 26], rounds=200, learning_rate=0.05, num_leaves=15, perturbation=0.05, seed=1
        ),
        synthetic_panel,
    )
    forecast = predict(model, synthetic_panel)
    assert (forecast >= 0).all()
    wape = (
        np.abs(forecast - synthetic_panel["units"].to_numpy()).sum()
        / synthetic_panel["units"].sum()
    )
    assert wape < 0.3
    # No tolerance check on the GBM elasticity: nothing forces a tree model to recover it,
    # which is the point of having it in the ladder. Only the sign.
    assert (own(model) <= 0.05).all()


@pytest.mark.parametrize("model_cls", [LogLog, CrossPrice])
def test_fit_never_sees_rows_it_is_not_given(synthetic_panel, model_cls):
    train = synthetic_panel.filter(
        pl.col("week_start") < pl.col("week_start").max() - pl.duration(weeks=20)
    )
    model = fit(model_cls(3, True), train)
    test = synthetic_panel.join(train, on=["store_id", "product_id", "week_start"], how="anti")
    assert len(predict(model, test)) == test.height
