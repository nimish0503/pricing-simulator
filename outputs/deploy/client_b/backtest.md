Rolling-origin backtest, Dominick's Finer Foods (cereal): 4 folds of 13 weeks, held-out weeks starting 1996-05-09 to 1997-05-01. Units are store x product x week units.

| Model | Client | WAPE all | WAPE promo | WAPE chain-week | Bias | MASE | Direction accuracy | Promo-lift calibration | Elasticity sd across folds | Plausibility |
|---|---|---|---|---|---|---|---|---|---|---|
| M0 naive | client_b | 58.7% | 82.4% | 43.8% | -43.3% | 1.00 | n/a | n/a | n/a | n/a |
| M1 log-log | client_b | 63.3% | 83.5% | 36.4% | 8.7% | 1.08 | 85.6% | 1.15 | 1.50 | 69.6% |
| M2 cross-price | client_b | 60.8% | 79.9% | 34.5% | 4.9% | 1.04 | 85.2% | 1.04 | 1.20 | 73.2% |
| M3 hierarchical | client_b | 59.8% | 79.3% | 33.4% | 3.2% | 1.02 | 85.3% | 1.01 | 1.18 | 74.1% |
| M4 LightGBM | client_b | 52.7% | 64.3% | 23.0% | -12.2% | 0.90 | 83.6% | 0.65 | 0.05 | 2.7% |
| Nested logit | client_b | 62.7% | 73.9% | 29.8% | -4.4% | 1.07 | 84.5% | 0.75 | 1.76 | 86.6% |
