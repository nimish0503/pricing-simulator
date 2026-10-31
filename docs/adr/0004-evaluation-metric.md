# ADR 0004: evaluation metric choice

Status: accepted.

## Context

Every model in the ladder is scored on the same rolling-origin folds: train to week t, forecast 13 weeks, step 13, at least 4 folds. The data has many low-volume and zero-sales store x product weeks, promos that multiply volume for one week, and two clients of very different size. One metric has to be the headline in the README results table, and the others back it up.

Forecast accuracy is half the story. A pricing model also needs elasticities that are stable across folds and plausible, so the evaluation needs at least one metric that isn't about forecast error.

## Options

Headline forecast metric:

1. WAPE, sum |actual - forecast| / sum actual. Volume-weighted, defined when individual weeks are zero, comparable across clients.
2. MAPE. Familiar to clients. Undefined on zero-sales weeks and dominated by tiny-volume rows.
3. RMSE on units. Punishes large misses, which are usually promo weeks. Scale-dependent, so not comparable across clients.
4. MASE against M0. Says directly whether a model beats naive. Less intuitive as a headline number for a client.

Pricing-quality metrics alongside it:

1. Elasticity stability: standard deviation of each product's elasticity across folds.
2. Plausibility rate: share of products inside the configured range.
3. Price-change direction accuracy on held-out weeks.
4. Promo-lift calibration: predicted vs actual lift on held-out promo weeks.

Also to decide: whether metrics are aggregated by pooling all rows or averaged per product, and how promo vs non-promo weeks are defined for the split.

## Decision

Headline: WAPE at store x product x week level, split into promo and non-promo weeks. It is defined when a week has zero sales, weights by volume, and is comparable across clients. MAPE is rejected because zero-sales weeks make it undefined and small-volume rows dominate it. MASE against M0 is reported next to it, so "beats naive" is one glance.

Also reported: bias, WAPE on chain-wide weekly category units (the error an optimizer gain on the category total is up against, and the band the optimizer reports), direction accuracy on held-out week-on-week price changes of 2% or more, promo-lift calibration against the M0 baseline, elasticity sd across folds, and plausibility rate.

Aggregation is by pooling all rows (ratio of sums), not averaging per product, so large products dominate as they do in revenue.

## Consequences

Store-level WAPE is high for every model (about 50-65% on client B) because single store-weeks are noisy and promo weeks are hard. That is normal for this grain and is why the chain-week WAPE is reported separately. Model choice for pricing uses the pricing-quality metrics, not WAPE alone (ADR 0003).
