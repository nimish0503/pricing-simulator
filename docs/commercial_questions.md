# Commercial questions

Fifteen questions a cereal pricing or category manager would bring to the first meeting. Each one names the output that answers it, so it's clear which phase has to exist before we can answer it.

## Price sensitivity

1. Which products are most price-sensitive, and which can take a price rise? (own-price elasticities, M1 to M3)
2. If we raise the regular price of our 12-pack by 5% for good, how many units do we lose per store per week? (regular-price elasticity, simulator)
3. Do shoppers in value stores react to price more than shoppers in upscale stores? (elasticity by store cluster)
4. What's the revenue-maximizing price for each product within a ±10% band? (optimizer, revenue objective)
5. Same question for gross margin: where do the revenue-best and margin-best prices disagree? (optimizer, margin objective)
6. Are we leaving money on the table at price thresholds, for example sitting at $3.09 when $2.99 would sell disproportionately more? (simulator around .x9 endings)

## Promotions

7. Does a feature ad add more units than a 20% price cut? (promo effects, promo planner)
8. Which promo mechanic gives the best incremental revenue per point of discount: price cut alone, feature, display, or a combination? (promo planner)
9. How much of a promo's lift is genuinely new volume, and how much is pulled from our own other products? (simulator, switching breakdown)
10. If we stop promoting a product in week 30, what happens to its units and to the rest of the category that week? (simulator scenario)
11. Is a temporary 20% cut a fair guide to what a permanent 20% cut would do? (regular vs promo elasticity)

## Portfolio and competition

12. If we cut the price of brand X, how much of its gain comes from our own brands versus competitors versus new category buyers? (cross-elasticities, nested logit)
13. Which products are close substitutes, so that pricing one means pricing the group? (product clusters, cross-elasticity heatmap)
14. Is our private label priced far enough below the national brand, or is the gap wide enough to give away margin? (price ladder, simulator)

## Confidence

15. How much should we trust these numbers: what's the forecast error on held-out weeks, and how stable is each elasticity from one period to the next? (backtest WAPE, elasticity stability)
