from datetime import date, timedelta

import numpy as np
import polars as pl

from shelfsim.config import Backtest, Split
from shelfsim.evaluation import metrics
from shelfsim.splits import final_fold, rolling_folds


def test_wape_bias_mase():
    actual = np.array([10.0, 0.0, 20.0])
    forecast = np.array([12.0, 1.0, 18.0])
    assert metrics.wape(actual, forecast) == 5 / 30
    assert metrics.bias(actual, forecast) == 1 / 30
    assert metrics.mase(actual, forecast, np.array([10.0, 10.0, 10.0])) == 5 / 20


def test_promo_lift_calibration():
    actual, naive = np.array([30.0, 40.0]), np.array([10.0, 10.0])
    assert metrics.promo_lift_calibration(actual, np.array([20.0, 30.0]), naive) == 30 / 50


def test_direction_accuracy_uses_consecutive_price_changes_only():
    weeks = [date(2020, 1, 6) + timedelta(weeks=t) for t in (0, 1, 2, 4)]
    frame = pl.DataFrame(
        {
            "store_id": ["1"] * 4,
            "product_id": ["a"] * 4,
            "week_start": weeks,
            "unit_price": [3.0, 2.5, 3.0, 2.5],
            "actual": [10.0, 20.0, 10.0, 30.0],
            "forecast": [10.0, 15.0, 12.0, 11.0],
        }
    )
    # Week 1: price down, actual up, forecast up (right). Week 2: price up, both down (right).
    # Week 4 follows a gap, so it doesn't count.
    assert metrics.direction_accuracy(frame, 0.02) == 1.0


def test_folds_are_ordered_in_time_and_never_overlap():
    sales = pl.DataFrame(
        {"week_start": [date(2010, 1, 4) + timedelta(weeks=t) for t in range(200)]}
    )
    split, bt = (
        Split(holdout_weeks=52, train_window_weeks=104),
        Backtest(horizon_weeks=13, step_weeks=13, n_folds=4, price_change_min=0.02),
    )
    final = final_fold(sales, split)
    assert final.train(sales)["week_start"].max() < final.test(sales)["week_start"].min()
    folds = rolling_folds(sales, split, bt)
    assert folds[0].train_end == final.train_end
    for fold in folds:
        assert fold.train(sales)["week_start"].max() < fold.test(sales)["week_start"].min()
        assert fold.test(sales).height == 13
