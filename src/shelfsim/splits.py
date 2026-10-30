"""Time splits. Every split is by date; nothing is ever shuffled."""

from dataclasses import dataclass
from datetime import date, timedelta

import polars as pl

from shelfsim.config import Backtest, Split


@dataclass(frozen=True)
class Fold:
    number: int
    train_start: date
    train_end: date
    test_end: date

    def train(self, sales: pl.DataFrame) -> pl.DataFrame:
        return sales.filter(pl.col("week_start").is_between(self.train_start, self.train_end))

    def test(self, sales: pl.DataFrame) -> pl.DataFrame:
        return sales.filter(
            (pl.col("week_start") > self.train_end) & (pl.col("week_start") <= self.test_end)
        )


def final_fold(sales: pl.DataFrame, split: Split) -> Fold:
    """Train on train_window_weeks ending holdout_weeks before the last week; test on the rest."""
    last = sales["week_start"].max()
    train_end = last - timedelta(weeks=split.holdout_weeks)
    return Fold(0, train_end - timedelta(weeks=split.train_window_weeks - 1), train_end, last)


def rolling_folds(sales: pl.DataFrame, split: Split, backtest: Backtest) -> list[Fold]:
    """Rolling origin over the holdout: train to t, test t+1..t+horizon, move t by step."""
    first_origin = final_fold(sales, split).train_end
    folds = []
    for i in range(backtest.n_folds):
        train_end = first_origin + timedelta(weeks=i * backtest.step_weeks)
        folds.append(
            Fold(
                i + 1,
                train_end - timedelta(weeks=split.train_window_weeks - 1),
                train_end,
                train_end + timedelta(weeks=backtest.horizon_weeks),
            )
        )
    return folds
