# ADR 0002: base-price estimation method

Status: accepted.

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

Option 1, same method for both clients: base price = max unit price over the trailing 8 weeks (current week included) of observed weeks without a promo code, forward-filled, and backward-filled at the start of a selling span. Client A's own BASE_PRICE is kept as `given_base_price` during onboarding only, and the data-quality report shows how often the two agree. A price this far under base (5%, `price_cut_min_depth`) counts as a price cut even without a code, because Dominick's doesn't code every deal (manual, part 4).

Rejected the mode (option 2): on Dominick's the regular price changes often enough that an 8-week mode has frequent ties. Rejected centred windows (option 3) because base price is a model input and must not use future weeks. Rejected option 4 because the two clients would no longer be measured the same way.

## Consequences

A permanent price cut is read as a temporary cut for up to 8 weeks, until the trailing max catches up. A permanent rise shows up at once. Both bias regular-price elasticity slightly toward zero around price changes. The 8-week window is in `configs/modeling.yaml`.
