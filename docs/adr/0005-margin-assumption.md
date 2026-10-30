# ADR 0005: margin assumption for client A

Status: proposed. The decision below is a draft to review; edit it into your own reasoning before marking it accepted.

## Context

Client B (Dominick's) reports PROFIT, the gross margin %, so `unit_cost = unit_price * (1 - PROFIT / 100)` can be derived per row. Client A (dunnhumby) reports no cost at all. The optimizer's margin objective, the promo planner's margin impact and the findings report all need a unit cost, so client A needs an assumption. Whatever it is, it lives in `configs/clients/client_a.yaml`, not in code, and the report says it's an assumption.

`unit_cost` is nullable in the schema, so one option is to leave it null and let margin outputs say "not available" for client A.

## Options

1. One flat gross margin for the whole category, for example a typical grocery cereal margin, applied to base price. Simple and easy to state. Ignores that private label usually carries a higher margin than national brands.
2. Margin by brand tier: one rate for national brands, a higher one for private label. Closer to reality, still two numbers in config.
3. Borrow client B's observed margins by tier and apply them to client A. Grounded in real data, but 1990s Chicago margins applied to a 2010s US retailer.
4. Leave cost null for client A and run only the revenue objective there. Honest, but loses half of the optimizer demo for one client.

Also to decide: whether cost is fixed per product or moves with base price, and whether promo funding from manufacturers lowers cost in promo weeks.

## Decision

Option 2 with option 3's numbers: margin on base price by brand tier, set to client B's median PROFIT in weeks 192-399 (17.5% national, 35.5% private label, from 2.26 million observed rows). Cost = base price x (1 - margin), fixed per week, so a temporary cut comes out of margin. Values live in `configs/clients/client_a.yaml`.

## Consequences

Margin numbers for client A are an assumption and the report says so. Using 1990s Chicago margins for a 2010s retailer is the weakest link; the private-label gap (twice the national margin) is the part most likely to carry over. No manufacturer promo funding is modelled for either client, so every promo pays for its own discount.
