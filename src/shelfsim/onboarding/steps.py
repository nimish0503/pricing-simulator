"""Onboarding steps shared by every client, run in this order:

1. unit_fixes       bundle prices to unit prices, revenue, cost and raw promo codes decoded
2. to_calendar      client weeks to week_start dates
   select_window_and_products   keep the analysis window and the products we model
3. fill_missing_weeks   store x product x week grid with observed / zero_sales rows
4. add_base_price   trailing max of non-promo prices
5. promo_flags      canonical flags and discount depth
6. flag_outliers    median + k * MAD per store-product, flagged and kept
7. validation and writing live in shelfsim.onboarding.run

Input columns are the intermediate names from the client YAML. Nothing here reads the client
id, so a new client is a YAML file.
"""

import re
from datetime import timedelta

import polars as pl

from shelfsim.config import ClientConfig, Onboarding, Selection, Sizes
from shelfsim.onboarding.quality import QualityLog

KEYS = ["store_id", "product_id"]


def _sql_flag(expr: str | None) -> pl.Expr:
    if expr is None:
        return pl.lit(None, dtype=pl.Boolean)
    return pl.sql_expr(expr).fill_null(False).cast(pl.Boolean)


def build_products(raw: pl.DataFrame, cfg: ClientConfig, log: QualityLog) -> pl.DataFrame:
    rules = cfg.product_rules
    frame = raw.unique("product_id", keep="first")
    if rules.category_filter:
        in_category = frame.filter(pl.sql_expr(rules.category_filter))
        log.drop_products(frame.join(in_category, on="product_id", how="anti"), "not in category")
        frame = in_category

    if "manufacturer" not in frame.columns:
        frame = frame.with_columns(manufacturer=_manufacturer_from_prefix(rules))
    if "category" not in frame.columns:
        frame = frame.with_columns(category=pl.lit(rules.category_name))

    frame = frame.with_columns(
        name=pl.col("name").cast(pl.String).str.strip_chars(),
        manufacturer=pl.col("manufacturer").cast(pl.String).str.strip_chars(),
        category=pl.lit(rules.category_name),
        size_oz=_size_oz(frame["product_id"], frame["size"], cfg.sizes),
    )
    frame = frame.with_columns(
        brand=pl.col("manufacturer"),
        is_private_label=_sql_flag(rules.private_label),
    )
    log.drop_products(frame.filter(pl.col("size_oz").is_null()), "size not parseable")
    frame = frame.filter(pl.col("size_oz").is_not_null())

    return frame.select(
        pl.lit(cfg.client_id).alias("client_id"),
        "product_id",
        "name",
        "brand",
        "manufacturer",
        "category",
        "size_oz",
        "is_private_label",
    )


def _manufacturer_from_prefix(rules) -> pl.Expr:
    expr = pl.lit("Other")
    # Shorter prefixes first, so a longer (more specific) prefix overrides them.
    for prefix, name in sorted(rules.manufacturer_prefixes.items(), key=lambda kv: len(kv[0])):
        expr = (
            pl.when(pl.col("product_id").str.starts_with(prefix)).then(pl.lit(name)).otherwise(expr)
        )
    return expr


def _size_oz(product_ids: pl.Series, sizes: pl.Series, rules: Sizes) -> pl.Series:
    units = sorted(rules.unit_to_oz, key=len, reverse=True)

    def parse(product_id: str, text: str | None) -> float | None:
        if product_id in rules.overrides:
            return rules.overrides[product_id]
        match = re.match(r"^\s*(\d+(?:\.\d+)?)\s*([A-Za-z]*)", str(text or ""))
        if not match:
            return None
        unit = match[2].lower() or rules.default_unit
        factor = next((rules.unit_to_oz[u] for u in units if unit.startswith(u)), None)
        return None if factor is None else float(match[1]) * factor

    return pl.Series(
        "size_oz", [parse(p, s) for p, s in zip(product_ids, sizes, strict=True)], dtype=pl.Float64
    )


def build_stores(raw: pl.DataFrame, cfg: ClientConfig) -> pl.DataFrame:
    rules = cfg.store_rules
    frame = raw.unique("store_id", keep="first")
    if rules.segment_map:
        frame = frame.with_columns(
            segment=pl.col("store_id").replace_strict(rules.segment_map, default=None)
        )
    for column in ("segment", "region"):
        if column not in frame.columns:
            frame = frame.with_columns(pl.lit(None, dtype=pl.String).alias(column))
    frame = frame.with_columns(
        _as_text("segment"),
        _as_text("region"),
        *(pl.col(c).cast(pl.Float64) for c in rules.extra_columns),
    )
    return frame.select(
        pl.lit(cfg.client_id).alias("client_id"),
        "store_id",
        "segment",
        "region",
        *rules.extra_columns,
    )


def _as_text(column: str) -> pl.Expr:
    text = pl.col(column).cast(pl.String).str.strip_chars()
    return (
        pl.when(text.str.contains(r"^\d+\.0+$"))
        .then(text.str.replace(r"\.0+$", ""))
        .otherwise(text)
        .alias(column)
    )


def unit_fixes(raw: pl.DataFrame, cfg: ClientConfig) -> pl.DataFrame:
    """Unit price per item, revenue, cost and decoded promo codes.

    A bundle price ("2 for $5") is the price of qty items, so unit price = price / qty.
    """
    frame = raw.with_columns(
        valid=pl.all_horizontal(pl.sql_expr(f) for f in cfg.row_filters)
        if cfg.row_filters
        else pl.lit(True)
    )
    qty = pl.col("qty").clip(lower_bound=1) if "qty" in frame.columns else pl.lit(1)
    frame = frame.with_columns(
        units=pl.col("units").cast(pl.Int64),
        unit_price=(pl.col("price").cast(pl.Float64) / qty),
    )
    revenue = (
        pl.col("revenue").cast(pl.Float64)
        if "revenue" in frame.columns
        else pl.col("unit_price") * pl.col("units")
    )
    unit_cost = pl.sql_expr(cfg.cost.unit_cost_expr) if cfg.cost.unit_cost_expr else pl.lit(None)
    return frame.with_columns(
        revenue=revenue,
        unit_cost=unit_cost.cast(pl.Float64),
        raw_feature=_sql_flag(cfg.promo.feature),
        raw_display=_sql_flag(cfg.promo.display),
        raw_price_cut=_sql_flag(cfg.promo.price_cut),
    )


def to_calendar(frame: pl.DataFrame, cfg: ClientConfig) -> pl.DataFrame:
    cal = cfg.calendar
    # CSV dates arrive as text, Excel and Stata dates as datetimes.
    as_date = (
        pl.col("week").str.to_date()
        if frame.schema["week"] == pl.String
        else pl.col("week").cast(pl.Date)
    )
    match cal.kind:
        case "week_number":
            week_start = pl.lit(cal.week1_start) + pl.duration(
                weeks=pl.col("week").cast(pl.Int64) - 1
            )
        case "week_end_date":
            week_start = as_date - pl.duration(days=6)
        case "week_start_date":
            week_start = as_date
    return frame.with_columns(week_start=week_start.cast(pl.Date))


def select_window_and_products(
    sales: pl.DataFrame, products: pl.DataFrame, rules: Selection, log: QualityLog
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Keep the last `last_weeks` weeks and the top products by revenue in that window."""
    last = sales["week_start"].max()
    start = last - timedelta(weeks=rules.last_weeks - 1)
    sales = sales.filter(pl.col("week_start") >= start)

    usable = pl.col("valid") & (pl.col("units") > 0) & (pl.col("unit_price") > 0)
    stats = (
        sales.filter(usable)
        .group_by("product_id")
        .agg(revenue=pl.col("revenue").sum(), weeks=pl.col("week_start").n_unique())
        .join(products.select("product_id", "is_private_label"), on="product_id", how="inner")
    )
    no_master = sales.select("product_id").unique().join(products, on="product_id", how="anti")
    log.drop_products(no_master, "no usable row in the product file")
    log.drop_products(
        products.join(stats, on="product_id", how="anti"), "no sales in the analysis window"
    )

    thin = stats.filter(pl.col("weeks") < rules.min_weeks_observed)
    log.drop_products(thin, f"sold in fewer than {rules.min_weeks_observed} weeks")
    ranked = stats.filter(pl.col("weeks") >= rules.min_weeks_observed).with_columns(
        rank=pl.col("revenue").rank("ordinal", descending=True).over("is_private_label")
    )
    keep = ranked.filter(
        pl.when(pl.col("is_private_label"))
        .then(pl.col("rank") <= rules.private_label_top_n)
        .otherwise(pl.col("rank") <= rules.top_n_products)
    )
    log.drop_products(
        ranked.join(keep, on="product_id", how="anti"),
        f"outside the top {rules.top_n_products} national / top {rules.private_label_top_n} "
        "private label products by revenue",
    )
    kept_ids = keep.select("product_id")
    return sales.join(kept_ids, on="product_id"), products.join(kept_ids, on="product_id")


def fill_missing_weeks(sales: pl.DataFrame, rules: Onboarding, log: QualityLog) -> pl.DataFrame:
    """Build the store x product x week grid and label every row.

    observed     units > 0 at a positive price on a valid row
    zero_sales   the store reported that week, the product sold before and after, and the run
                 of zero weeks is at most max_zero_run_weeks long
    not_carried  zero runs longer than that, or weeks outside the product's selling span
                 in that store. Dropped.
    no data      the store has no valid row for any product that week, or this row failed a
                 client row filter (Dominick's OK = 0). Dropped.
    """
    usable = pl.col("valid") & (pl.col("units") > 0) & (pl.col("unit_price") > 0)
    observed = sales.filter(usable)
    invalid = sales.filter(~pl.col("valid")).select(*KEYS, "week_start").unique()

    # A store reported a week if it has any valid row that week, including an explicit zero.
    store_weeks = sales.filter(pl.col("valid")).select("store_id", "week_start").unique()
    spans = observed.group_by(KEYS).agg(
        first=pl.col("week_start").min(), last=pl.col("week_start").max()
    )
    grid = (
        spans.join(store_weeks, on="store_id")
        .filter(pl.col("week_start").is_between(pl.col("first"), pl.col("last")))
        .join(invalid, on=[*KEYS, "week_start"], how="anti")
        .select(*KEYS, "week_start")
    )
    carried = [c for c in sales.columns if c not in (*KEYS, "week_start")]
    frame = grid.join(
        observed.select(*KEYS, "week_start", *carried), on=[*KEYS, "week_start"], how="left"
    ).with_columns(
        row_status=pl.when(pl.col("units").is_not_null())
        .then(pl.lit("observed"))
        .otherwise(pl.lit("zero_sales"))
    )

    frame = frame.sort(*KEYS, "week_start").with_columns(
        run=(pl.col("row_status") != pl.col("row_status").shift())
        .fill_null(True)
        .cum_sum()
        .over(KEYS)
    )
    frame = frame.with_columns(run_length=pl.len().over(*KEYS, "run"))
    off_shelf = (pl.col("row_status") == "zero_sales") & (
        pl.col("run_length") > rules.max_zero_run_weeks
    )

    log.step(
        "missing weeks",
        frame.height,
        f"{(frame['row_status'] == 'zero_sales').sum():,} grid rows without a sale; "
        f"{frame.filter(off_shelf).height:,} of them are in zero runs longer than "
        f"{rules.max_zero_run_weeks} weeks (not_carried, dropped); "
        f"{invalid.height:,} rows failed a row filter (no data, dropped)",
    )
    frame = frame.filter(~off_shelf).drop("run", "run_length")
    zero = pl.col("row_status") == "zero_sales"
    return frame.with_columns(
        units=pl.when(zero).then(0).otherwise(pl.col("units")),
        revenue=pl.when(zero).then(0.0).otherwise(pl.col("revenue")),
        valid=pl.lit(True),
        **{
            c: pl.when(zero).then(False).otherwise(pl.col(c))
            for c in ("raw_price_cut", "raw_feature", "raw_display")
        },
    )


def add_base_price(sales: pl.DataFrame, rules: Onboarding) -> pl.DataFrame:
    """Regular price = max unit price over the trailing window of non-promo observed weeks.

    Promo weeks don't count, so a temporary cut doesn't drag the base price down. If a
    store-product has no non-promo week yet, the first later one is used, and failing that the
    max of all its observed prices. Zero-sales weeks get the base price as their unit price,
    since nothing was sold at any other price.
    """
    window = f"{7 * rules.base_price_window_weeks}d"
    regular = pl.when((pl.col("row_status") == "observed") & ~pl.col("raw_price_cut")).then(
        pl.col("unit_price")
    )
    frame = sales.sort(*KEYS, "week_start").with_columns(
        base_price=regular.rolling_max_by("week_start", window_size=window).over(KEYS)
    )
    frame = frame.with_columns(
        base_price=pl.col("base_price")
        .forward_fill()
        .backward_fill()
        .fill_null(pl.col("unit_price").max())
        .over(KEYS)
    )
    return frame.with_columns(
        unit_price=pl.when(pl.col("row_status") == "zero_sales")
        .then(pl.col("base_price"))
        .otherwise(pl.col("unit_price"))
    )


def promo_flags(sales: pl.DataFrame, rules: Onboarding) -> pl.DataFrame:
    """discount_depth = 1 - unit_price / base_price, clipped at 0 for weeks priced above base.

    A week is a price cut if the client's code says so or the depth reaches the threshold,
    because some clients (Dominick's) don't code every promo.
    """
    depth = (1 - pl.col("unit_price") / pl.col("base_price")).clip(lower_bound=0.0)
    return sales.with_columns(
        price_above_base=pl.col("unit_price") > pl.col("base_price") * 1.0001,
        discount_depth=depth,
        on_feature=pl.col("raw_feature"),
        on_display=pl.col("raw_display"),
        on_price_cut=(pl.col("raw_price_cut") | (depth >= rules.price_cut_min_depth)),
    )


def flag_outliers(sales: pl.DataFrame, rules: Onboarding) -> pl.DataFrame:
    """Flag regular-price weeks with units above median + k * MAD of the store-product's
    regular-price weeks.

    Promo weeks are never flagged: a deep cut selling five times the usual volume is the signal
    the models need, and promo depth varies too much for one threshold to separate a spike from
    a good promo. MAD is floored at one unit, so a product that sells exactly 3 a week every
    week doesn't get every 4 flagged.
    """
    regular = (pl.col("row_status") == "observed") & ~pl.col("on_price_cut")
    frame = sales.with_columns(
        median_units=pl.col("units").filter(regular).cast(pl.Float64).median().over(KEYS)
    )
    mad = (
        (pl.col("units") - pl.col("median_units"))
        .abs()
        .filter(regular)
        .median()
        .over(KEYS)
        .clip(lower_bound=1.0)
    )
    return frame.with_columns(
        is_outlier=(
            regular & (pl.col("units") > pl.col("median_units") + rules.outlier_mad_k * mad)
        ).fill_null(False)
    ).drop("median_units")


def add_unit_cost(sales: pl.DataFrame, products: pl.DataFrame, cfg: ClientConfig) -> pl.DataFrame:
    """Cost from the client's own data where it has it, otherwise from the margin assumption.

    Zero-sales weeks get the store-product's most recent known cost.
    """
    if cfg.cost.margin_national is not None:
        frame = sales.join(products.select("product_id", "is_private_label"), on="product_id")
        margin = (
            pl.when(pl.col("is_private_label"))
            .then(cfg.cost.margin_private_label)
            .otherwise(cfg.cost.margin_national)
        )
        return frame.with_columns(unit_cost=pl.col("base_price") * (1 - margin)).drop(
            "is_private_label"
        )
    return sales.sort(*KEYS, "week_start").with_columns(
        unit_cost=pl.col("unit_cost").forward_fill().backward_fill().over(KEYS)
    )
