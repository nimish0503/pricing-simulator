# ADR 0002: base-price estimation method

Status: proposed

## Context

Elasticity of regular price and response to a promo are different things, and separating them needs a regular (base) price for every row. Client A ships BASE_PRICE. Client B doesn't. To keep the two clients comparable, the same estimation method has to run on both, which means client A's given BASE_PRICE becomes a check on the method, not an input.

The method also feeds `discount_depth = 1 - unit_price / base_price`, so it decides which weeks count as promo weeks wherever a client has no explicit price-cut flag.

## Options

1. Trailing max of non-promo prices over 8 weeks. Simple. Fast to react when the regular price goes up, but it jumps up on any one-week data error, and it lags by up to 8 weeks when the regular price goes down.
2. Trailing mode over 8 weeks. Regular prices repeat, promo prices mostly don't, so the mode picks the shelf price without needing promo flags. Ties and short histories need a rule.
3. Centered rolling max or mode (looks at future weeks too). Cleaner breakpoints, but it uses information from after the week, which is fine for describing history and wrong if the base price is ever used as a forecasting feature.
4. Use client-given base price where it exists, estimate only where it doesn't. Uses the best data per client, at the cost of the two clients no longer being measured the same way.

Parameters to set for whichever option wins: window length, minimum weeks of history, and what counts as "non-promo" when a client has no flags.

## Decision

## Consequences
