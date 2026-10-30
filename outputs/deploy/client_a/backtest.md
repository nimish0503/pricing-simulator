Rolling-origin backtest, dunnhumby Breakfast at the Frat (cold cereal): 4 folds of 13 weeks, held-out weeks starting 2011-01-06 to 2011-12-29. Units are store x product x week units.

| Model | Client | WAPE all | WAPE promo | WAPE chain-week | Bias | MASE | Direction accuracy | Promo-lift calibration | Elasticity sd across folds | Plausibility |
|---|---|---|---|---|---|---|---|---|---|---|
| M0 naive | client_a | 42.3% | 58.6% | 24.1% | -22.3% | 1.00 | n/a | n/a | n/a | n/a |
| M1 log-log | client_a | 32.5% | 35.5% | 8.7% | -0.6% | 0.77 | 68.4% | 1.07 | 0.36 | 85.0% |
| M2 cross-price | client_a | 33.4% | 37.4% | 10.2% | 0.6% | 0.79 | 68.6% | 1.12 | 0.33 | 90.0% |
| M3 hierarchical | client_a | 32.7% | 35.7% | 9.1% | 0.0% | 0.77 | 68.5% | 1.08 | 0.41 | 81.7% |
| M4 LightGBM | client_a | 31.1% | 33.3% | 9.2% | 0.1% | 0.74 | 69.6% | 1.05 | 0.38 | 53.3% |
| Nested logit | client_a | 53.1% | 57.8% | 14.5% | 2.4% | 1.25 | 66.9% | 1.50 | 0.22 | 100.0% |
