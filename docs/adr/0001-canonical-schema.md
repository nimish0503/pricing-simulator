# ADR 0001: canonical schema design

Status: accepted.

## Context

Two clients send scanner data in different shapes. Client A (dunnhumby) has per-unit PRICE and BASE_PRICE and three promo flags. Client B (Dominick's) has bundle prices (PRICE for QTY units), a SALE code instead of flags, a PROFIT margin, and missing rows where a store didn't report. Everything after onboarding has to run on both without knowing which is which, so the schema decides what every downstream model can assume.

The starting point from the spec, as implemented in `src/shelfsim/schema.py`:

| Table | Grain | Columns |
|---|---|---|
| sales | client x store x product x week | client_id, store_id, product_id, week_start, units, revenue, unit_price, base_price, discount_depth, on_feature, on_display, on_price_cut, unit_cost (nullable) |
| products | client x product | client_id, product_id, name, brand, manufacturer, category, size_oz, is_private_label |
| stores | client x store | client_id, store_id, segment, region, plus optional demographic columns |

Questions this ADR has to settle:

- How a filled-in missing week is represented. Phase 2 labels rows observed, zero_sales or not_carried, and the current schema has no column for that, and requires `unit_price > 0` on every row, which a not_carried row may not have.
- What `discount_depth` does when the actual price is above the estimated base price. The schema currently requires `0 <= discount_depth < 1`, so onboarding would have to clip.
- Whether price columns are per unit only, or whether price per ounce is stored too.
- Whether promo flags stay as three booleans or become one categorical mechanic.
- Whether the stores table is strict about columns. It isn't now, so demographic columns pass through.

## Options

Row status for filled weeks:

1. Add a `row_status` column (observed, zero_sales, not_carried) to sales and allow null prices on non-observed rows.
2. Keep sales observed-only and store the filled grid in a separate table.
3. Drop not_carried rows before writing and keep zero_sales rows with the base price carried forward.

Discount depth above base price:

1. Clip to 0 and count clipped rows in the data-quality report.
2. Allow negative values and let models see price increases above the regular price.

Price per ounce:

1. Store `price_per_oz` in sales.
2. Derive it on read from `unit_price / products.size_oz`.

Promo representation:

1. Three booleans plus `discount_depth`, as now.
2. One categorical `promo_mechanic` column plus `discount_depth`.

## Decision

Row status: option 3, plus a column. `sales` holds observed and zero_sales rows only, labelled by a `row_status` column. Weeks outside a store-product's selling span, zero runs longer than `max_zero_run_weeks` (8) and rows failing a client row filter are dropped and counted in the data-quality report. A zero_sales row gets the base price as its unit price, so `unit_price > 0` holds on every row. Rejected option 1 (null prices) because every model would need a null-price branch; rejected option 2 (separate grid table) because the zero weeks matter to WAPE and belong next to the sales.

Discount depth: option 1, clip at 0. 189 of 403,982 client B rows are priced above base and the report counts them. Negative depth would let a model read a price rise above "regular" as an anti-promo, which the data is too thin to support.

Price per ounce: option 2, derived on read from `unit_price / products.size_oz`. One source of truth for size.

Promo representation: option 1, three booleans plus depth. `on_feature` and `on_display` are nullable: null means the client doesn't report the signal (Dominick's), which is different from "not featured". Models treat null as 0 and drop a regressor with no variation.

Added `is_outlier` (flagged, never deleted) so models can choose to exclude outliers from fitting.

Stores are not strict: client demographic columns pass through to Parquet. In DuckDB they go to a long `store_attributes` table so the `stores` table has the same columns for every client.

## Consequences

Downstream code never needs to know whether a client reports feature or display. A client whose promo data has more mechanics (coupon, bonus buy) loses that detail: Dominick's B, C and S codes all become `on_price_cut`. The fake third client test (`tests/onboarding/test_onboarding.py`) shows a new format needs only a YAML file.
