# ADR 0006: market size for the choice model

Status: accepted.

## Context

The logit share model treats each store-week as a market of M shoppers who each buy one product or nothing. Shares are s_j = units_j / M and the outside share is s_0 = 1 - sum s_j, so M decides how big the outside option is. That matters more than it looks: in a logit the category-level price response is roughly alpha x price x s_0, so a large outside share makes the whole category look price-elastic.

## Options

1. Store traffic times a penetration rate. Grounded in shopper counts, but needs traffic data in the same form for every client (Dominick's has daily customer counts, dunnhumby only an average weekly basket count) and a penetration rate nobody measured.
2. A multiple of the store's peak weekly category units in training. Needs only the sales table, so it works the same for every client, and guarantees s_0 > 0 in every week seen so far.

## Decision

Option 2 with a multiple of 2 (`market_peak_multiple` in `configs/modeling.yaml`). It needs no data beyond the canonical tables, which keeps the choice model client-agnostic.

## Consequences

On client B the outside share is about 75% in a typical week, and the logit's implied category elasticity is about -2. That is too elastic for cereal as a whole: maximizing revenue, the optimizer cuts every price toward the bottom of its band, because it believes lost category buyers come back easily. Maximizing margin it raises prices and the volume floor binds, which is the more credible answer. So margin is the default objective, and the report shows the revenue answer only as a comparison. A smaller multiple (say 1.3) would make the category less elastic; testing it against the regression models' category response is the obvious next step.
