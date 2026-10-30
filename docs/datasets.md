# Datasets: what the raw files actually contain

Checked against the files and the Dominick's data manual. Client A's details are still from the dunnhumby user guide, because the raw file isn't downloaded yet.

## Client B: Dominick's Finer Foods, cereal

Source: Kilts Center for Marketing, Chicago Booth. Files used: `wcer.csv` (movement, 6,602,582 rows), `upccer.csv` (490 UPCs), `demo.dta` (store demographics), and the data manual. Free for research use; the raw files are not redistributed here.

| Field | What it is | How onboarding uses it |
|---|---|---|
| STORE, UPC, WEEK | keys; week 1 starts 1989-09-14 and weeks are consecutive 7-day periods | `week_start = 1989-09-14 + (WEEK - 1) weeks` |
| MOVE | units sold, counted as items, not bundles | `units` |
| QTY, PRICE | PRICE is for a bundle of QTY items; 5,034 rows have QTY > 1 | `unit_price = PRICE / QTY` |
| SALE | B bonus buy, C coupon, S price reduction; the manual says it is not set consistently | any code sets `on_price_cut`; so does a price 5%+ under base |
| PROFIT | gross margin % on average acquisition cost | `unit_cost = unit_price x (1 - PROFIT / 100)` when 0 < PROFIT < 100 |
| OK | 0 marks a week the Kilts Center considers suspect (141,285 rows) | treated as no data |

Things that differ from the spec's description:

- Zero-sales weeks are mostly explicit rows with MOVE = 0 and PRICE = 0, not missing rows. Both are handled the same way: inside the selling span they become `zero_sales` rows.
- There are no feature or display flags in the movement file.
- The whole chain has no data for weeks 284-309 (26 weeks, 1995-02-16 to 1995-08-10), 262-265 and 370-371, and the analysis window has one more missing week in 1993.
- Brand isn't a field. The manufacturer comes from the UPC prefix (16000 General Mills, 38000 Kellogg's, 43000 Post, 30000 Quaker, 38281 Dominick's own label), and DESCRIP is cut at 20 characters.
- SIZE is free text ("15 OZ", "23.45O", "ASST", "end"). One top seller, Crispix, has SIZE "end" and is set to 12 oz in the YAML.
- Store price tier (High, Medium, Low, CubFighter) isn't in `demo.dta`; it's in the manual's store table, and is copied into `configs/clients/client_b.yaml` for the 76 stores listed there.

## Client A: dunnhumby "Breakfast at the Frat"

Not downloaded yet. `configs/clients/client_a.yaml` maps the sheet and column names from dunnhumby's user guide: transactions (WEEK_END_DATE, STORE_NUM, UPC, UNITS, SPEND, PRICE, BASE_PRICE, FEATURE, DISPLAY, TPR_ONLY), products (UPC, DESCRIPTION, MANUFACTURER, CATEGORY, PRODUCT_SIZE) and stores (STORE_ID, SEG_VALUE_NAME, ADDRESS_STATE_PROV_CODE, AVG_WEEKLY_BASKETS). The adapter finds the header row in each sheet, so a title block above it is fine. `tests/onboarding/test_onboarding.py` runs a workbook in that format through the real config. When the real file arrives, check the names against it and fix the YAML, not the code.
