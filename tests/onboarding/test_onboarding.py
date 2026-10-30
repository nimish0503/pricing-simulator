from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pytest

from shelfsim import schema
from shelfsim.config import Onboarding, load_client
from shelfsim.onboarding import onboard, steps
from shelfsim.onboarding.quality import QualityLog

FIXTURES = Path(__file__).parents[1] / "fixtures" / "configs"
RULES = Onboarding(
    base_price_window_weeks=8, max_zero_run_weeks=2, price_cut_min_depth=0.05, outlier_mad_k=5.0
)


def grid_rows(units, prices, raw_cut=None, valid=None, store="1", product="a"):
    n = len(units)
    return pl.DataFrame(
        {
            "store_id": [store] * n,
            "product_id": [product] * n,
            "week_start": [date(2020, 1, 6) + timedelta(weeks=t) for t in range(n)],
            "units": units,
            "unit_price": prices,
            "revenue": [u * p for u, p in zip(units, prices, strict=True)],
            "valid": valid or [True] * n,
            "raw_price_cut": raw_cut or [False] * n,
            "raw_feature": [False] * n,
            "raw_display": [False] * n,
            "unit_cost": [None] * n,
        },
        schema_overrides={"unit_cost": pl.Float64},
    )


def test_bundle_price_becomes_unit_price():
    cfg = load_client("client_b")
    raw = pl.DataFrame(
        {
            "store_id": ["2"],
            "product_id": ["1"],
            "week": [1],
            "units": [6],
            "price": [5.0],
            "qty": [2],
            "sale": ["B"],
            "profit": [20.0],
            "ok": [1],
        }
    )
    out = steps.unit_fixes(raw, cfg)
    assert out["unit_price"][0] == 2.5
    assert out["revenue"][0] == 15.0
    assert out["unit_cost"][0] == pytest.approx(2.0)
    assert out["raw_price_cut"][0]
    assert steps.to_calendar(out, cfg)["week_start"][0] == date(1989, 9, 14)


def test_missing_weeks_are_labelled_and_long_gaps_dropped():
    # Store 1 sells in weeks 0, 1, 3, 7: week 2 is a one-week gap (zero_sales), weeks 4-6 are a
    # three-week gap (longer than max_zero_run_weeks = 2, so not carried). Store 2 reports
    # every week, so every week counts as a reporting week.
    a = grid_rows([5, 4, 0, 6, 0, 0, 0, 5], [2.0] * 8)
    b = grid_rows([3] * 8, [2.0] * 8, store="2")
    out = steps.fill_missing_weeks(pl.concat([a, b]), RULES, QualityLog()).filter(
        pl.col("store_id") == "1"
    )
    assert out["row_status"].to_list() == [
        "observed",
        "observed",
        "zero_sales",
        "observed",
        "observed",
    ]
    assert out["units"].to_list() == [5, 4, 0, 6, 5]


def test_rows_failing_a_row_filter_are_no_data_not_zero_sales():
    a = grid_rows([5, 4, 3, 6], [2.0] * 4, valid=[True, False, True, True])
    out = steps.fill_missing_weeks(a, RULES, QualityLog())
    assert out.height == 3
    assert (out["row_status"] == "observed").all()


def test_base_price_ignores_promo_weeks_and_depth_is_computed():
    prices = [3.0, 3.0, 2.4, 3.0, 3.2, 3.2, 3.2, 2.88]
    frame = steps.fill_missing_weeks(
        grid_rows([5] * 8, prices, raw_cut=[False, False, True] + [False] * 5), RULES, QualityLog()
    )
    frame = steps.promo_flags(steps.add_base_price(frame, RULES), RULES)
    assert frame["base_price"].to_list() == [3.0, 3.0, 3.0, 3.0, 3.2, 3.2, 3.2, 3.2]
    assert frame["discount_depth"][2] == pytest.approx(0.2)
    # Week 7 has no promo code, but 10% under base counts as a price cut by the depth rule.
    assert frame["on_price_cut"].to_list() == [False, False, True, False, False, False, False, True]


def test_outliers_flagged_on_regular_weeks_only():
    units = [10, 11, 9, 10, 200, 60, 10, 10]
    cut = [False] * 5 + [True] + [False] * 2
    frame = steps.fill_missing_weeks(grid_rows(units, [2.0] * 8, raw_cut=cut), RULES, QualityLog())
    frame = steps.flag_outliers(steps.promo_flags(steps.add_base_price(frame, RULES), RULES), RULES)
    assert frame["is_outlier"].to_list() == [False, False, False, False, True, False, False, False]


def test_size_parsing():
    cfg = load_client("client_b")
    sizes = steps._size_oz(
        pl.Series(["1", "2", "3", "4", "3800003500"]),
        pl.Series(["15 OZ", "23.45O", "1.5 LB", "ASST", "end"]),
        cfg.sizes,
    )
    assert sizes.to_list() == [15.0, 23.45, 24.0, None, 12.0]


def write_fake_client(root: Path) -> None:
    raw = root / "raw" / "client_c"
    raw.mkdir(parents=True)
    rng = np.random.default_rng(0)
    rows = []
    for shop in ("S1", "S2", "S3"):
        for sku, price in (("A", 2.5), ("B", 3.0), ("C", 1.8), ("D", 4.0)):
            for t in range(30):
                promo = ["", "TPR", "FEAT", "DISP"][t % 7] if t % 7 < 4 and t % 3 == 0 else ""
                shelf = price * (0.8 if promo == "TPR" else 1.0)
                qty = int(rng.poisson(20 if not promo else 40))
                if sku == "D" and t < 25:
                    continue  # too few weeks, should be dropped
                rows.append(
                    {
                        "shop": shop,
                        "sku": sku,
                        "wk_ending": str(date(2021, 1, 9) + timedelta(weeks=t)),
                        "qty_sold": qty,
                        "sales_value": qty * shelf,
                        "shelf_price": shelf,
                        "promo_type": promo,
                    }
                )
    pd.DataFrame(rows).to_csv(raw / "movement.csv", index=False)
    pd.DataFrame(
        {
            "sku": ["A", "B", "C", "D"],
            "desc": ["OAT RINGS", "CORN FLAKES", "OWN OAT RINGS", "GRANOLA"],
            "brand_owner": ["MillCo", "Flakeco", "OWN LABEL", "MillCo"],
            "pack": ["375g", "500g", "375g", "1kg"],
        }
    ).to_csv(raw / "items.csv", index=False)
    pd.DataFrame(
        {"shop": ["S1", "S2", "S3"], "banner": ["express", "superstore", "superstore"]}
    ).to_csv(raw / "shops.csv", index=False)


def test_fake_third_client_needs_only_yaml(tmp_path, mcfg):
    write_fake_client(tmp_path)
    cfg = load_client("client_c", FIXTURES)
    tables = onboard(
        cfg,
        mcfg,
        root=tmp_path,
        out_dir=tmp_path,
        canon_dir=tmp_path / "canonical",
        db_path=tmp_path / "test.duckdb",
    )
    schema.sales.validate(tables["sales"])
    products = tables["products"].sort("product_id")
    assert products["product_id"].to_list() == ["A", "B", "C"]
    assert products["size_oz"].to_list() == pytest.approx(
        [375 * 0.035274, 500 * 0.035274, 375 * 0.035274]
    )
    assert products["is_private_label"].to_list() == [False, False, True]
    sales = tables["sales"]
    assert (
        sales.filter(pl.col("on_feature")).height > 0
        and sales.filter(pl.col("on_display")).height > 0
    )
    assert (tmp_path / "data_quality.html").exists()
    import duckdb

    with duckdb.connect(str(tmp_path / "test.duckdb")) as con:
        assert (
            con.execute("SELECT count(*) FROM sales WHERE client_id = 'client_c'").fetchall()[0][0]
            == sales.height
        )


def write_dunnhumby_workbook(path: Path) -> None:
    weeks = [date(2009, 1, 17) + timedelta(weeks=t) for t in range(60)]
    tx = [
        [
            "WEEK_END_DATE",
            "STORE_NUM",
            "UPC",
            "UNITS",
            "VISITS",
            "HHS",
            "SPEND",
            "PRICE",
            "BASE_PRICE",
            "FEATURE",
            "DISPLAY",
            "TPR_ONLY",
        ]
    ]
    for w_i, w in enumerate(weeks):
        for store in (367, 389):
            for upc, price in ((1111009477, 3.19), (1600027527, 3.49), (1111085319, 2.49)):
                cut = w_i % 6 == 0
                p = round(price * (0.8 if cut else 1.0), 2)
                units = 30 if cut else 12
                tx.append(
                    [w, store, upc, units, units, units, units * p, p, price, int(cut), 0, int(cut)]
                )
    products = [
        ["UPC", "DESCRIPTION", "MANUFACTURER", "CATEGORY", "SUB_CATEGORY", "PRODUCT_SIZE"],
        [1111009477, "PL MINI TWIST PRETZELS", "PRIVATE LABEL", "BAG SNACKS", "PRETZELS", "15 OZ"],
        [
            1600027527,
            "GM HONEY NUT CHEERIOS",
            "GENERAL MI",
            "COLD CEREAL",
            "ALL FAMILY CEREAL",
            "12.25 OZ",
        ],
        [
            1111085319,
            "PL HONEY NUT TOASTD OATS",
            "PRIVATE LABEL",
            "COLD CEREAL",
            "ALL FAMILY CEREAL",
            "12.25 OZ",
        ],
    ]
    stores = [
        [
            "STORE_ID",
            "STORE_NAME",
            "ADDRESS_CITY_NAME",
            "ADDRESS_STATE_PROV_CODE",
            "MSA_CODE",
            "SEG_VALUE_NAME",
            "PARKING_SPACE_QTY",
            "SALES_AREA_SIZE_NUM",
            "AVG_WEEKLY_BASKETS",
        ],
        [367, "SALEM", "SALEM", "IN", 99999, "VALUE", None, 24721, 12707],
        [389, "MIDDLETOWN", "MIDDLETOWN", "OH", 17140, "MAINSTREAM", 408, 46073, 24767],
    ]
    with pd.ExcelWriter(path) as writer:
        for sheet, body in (
            ("dh Transaction Data", tx),
            ("dh Products Lookup", products),
            ("dh Store Lookup", stores),
        ):
            width = len(body[0])
            title = [["dunnhumby - Breakfast at the Frat"] + [None] * (width - 1), [None] * width]
            pd.DataFrame(title + body).to_excel(writer, sheet_name=sheet, header=False, index=False)


def test_client_a_workbook_format(tmp_path, mcfg):
    raw = tmp_path / "data" / "raw" / "client_a"
    raw.mkdir(parents=True)
    write_dunnhumby_workbook(raw / "dunnhumby - Breakfast at the Frat.xlsx")
    cfg = load_client("client_a")
    tables = onboard(
        cfg, mcfg, root=tmp_path, out_dir=tmp_path, canon_dir=tmp_path / "canonical", db_path=None
    )
    assert set(tables["products"]["product_id"]) == {"1600027527", "1111085319"}  # pretzels dropped
    sales = tables["sales"]
    assert sales.filter(pl.col("on_feature")).height > 0
    assert tables["stores"]["segment"].to_list() in (
        ["VALUE", "MAINSTREAM"],
        ["MAINSTREAM", "VALUE"],
    )
    assert sales["unit_cost"].null_count() == 0
