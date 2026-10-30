# ADR 0003: demand model family

Status: proposed

## Context

The models have two jobs that pull in different directions: forecast weekly units per store x product, and give elasticities a pricing manager can act on. A model can do the first well and the second badly. The spec's ladder is M0 naive, M1 log-log OLS, M2 pooled with cross-prices, M3 hierarchical Bayes and M4 LightGBM, all behind one `fit` / `predict` interface. This ADR records which of them is the model of record for pricing advice and why.

Things that have to hold for the chosen model:

- It separates regular-price elasticity from promo response.
- It produces cross-price elasticities within substitution groups.
- It gives an interval for each elasticity.
- It handles products with few weeks or little price variation without wild estimates.

## Options

1. Log-log OLS per product (M1). Elasticity is the coefficient on ln(base price), with standard errors from statsmodels. Easy to explain. Thin products get noisy estimates and there are no cross-prices.
2. Pooled log-log with cross-prices per product cluster (M2). Adds cannibalization. Pooling assumes products in a cluster share the promo and seasonal response.
3. Hierarchical Bayes with partial pooling toward a cluster mean (M3). Stable estimates for thin products and credible intervals. Slower to fit and harder to explain to a client in one sentence.
4. Gradient boosting (M4). Probably the best forecaster. Elasticity only comes out by perturbing price, and nothing forces it to be negative or smooth.
5. Treat price as endogenous and estimate with Hausman instruments and 2SLS. Addresses the bias from retailers setting prices in response to demand. Adds a dependency (linearmodels) and needs the same product sold in many stores.

Also to decide: functional form (log-log constant elasticity vs semi-log), and whether elasticities vary by store cluster.

## Decision

## Consequences
