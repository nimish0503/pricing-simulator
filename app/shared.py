"""Loaders and page components shared by the app. The app reads only outputs/deploy/, never raw data."""

import html
import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import polars as pl
import streamlit as st

from shelfsim.choice.simulator import Scenario, Simulator
from shelfsim.config import load_modeling

DEPLOY = Path(__file__).resolve().parents[1] / "outputs" / "deploy"
SETTINGS = load_modeling()
PLAUSIBLE = (SETTINGS.plausibility.own_elasticity_min, SETTINGS.plausibility.own_elasticity_max)

# Mid-tone colors that keep 3:1 contrast on both the light and the dark background.
BLUE = "#2563EB"
AMBER = "#D97706"
GREEN = "#15803D"
RED = "#DC2626"
GREY = "#94A3B8"
SLATE = "#64748B"
MAKER_PALETTE = ["#2563EB", "#D97706", "#0F766E", "#7C3AED", "#DB2777", "#0891B2", "#65A30D"]

MODELS = {
    "M1 log-log": "Simple regression",
    "M2 cross-price": "Regression with competitor prices",
    "M3 hierarchical": "Hierarchical Bayes (recommended)",
    "M4 LightGBM": "Machine learning (LightGBM)",
    "Nested logit": "Shopper choice model",
}

GLOSSARY = {
    "Price elasticity": "The % change in weekly units when the regular price rises 1%. "
    "-2 means a 1% price rise loses 2% of units.",
    "Expected range": f"{PLAUSIBLE[0]:g} to {PLAUSIBLE[1]:g}. Cereal demand falls when price "
    f"rises, but rarely by more than {-PLAUSIBLE[0]:g}% per 1%. Estimates outside it are "
    "probably noise.",
    "Temporary-cut elasticity": "The same idea for a one-off promo price cut. Shoppers stock "
    "up on deals, so this is usually bigger than the regular-price elasticity.",
    "Cross elasticity": "The % change in one product's units when another product's price "
    "rises 1%. Positive means shoppers switch between the two.",
    "Gross margin": "Revenue minus product cost. For dunnhumby the cost is assumed: 17.5% "
    "margin on national brands, 35.5% on private label.",
    "Feature": "The product is in the weekly ad or circular.",
    "Display": "The product has an in-store display, like an end cap.",
    "Price cut week": "A week where the shelf price is at least 5% below the regular price.",
    "Forecast error band": "How far off the models' weekly category forecasts were on a year "
    "of held-out data. A gain smaller than this band could be noise.",
    "Product cluster": "Products grouped by price tier, pack size and promo behaviour. "
    "Shoppers mostly switch within a cluster.",
}

PROMOS = {
    "10% price cut": "10% off",
    "20% price cut": "20% off",
    "feature only": "Weekly ad",
    "display only": "In-store display",
    "feature + display + 10% cut": "Ad + display + 10% off",
}

CSS = """
<style>
.block-container { padding-top: 4rem; max-width: 1280px; }
.page-title { font-size: 2rem; font-weight: 700; line-height: 1.2; margin: 0 0 .5rem; padding: 0; }
.lede { font-size: 1.02rem; opacity: .78; max-width: 52rem; margin: 0 0 .6rem; }
[data-testid="stMetricValue"], [data-testid="stDataFrame"] { font-variant-numeric: tabular-nums; }
[data-testid="stMetricLabel"] p { font-weight: 500; }
h2 { margin-top: .8rem; }
.hero {
  border-radius: 14px; padding: 26px 30px 24px; color: #fff; margin: .2rem 0 .6rem;
  background: linear-gradient(120deg, #1E3A8A 0%, #1D4ED8 55%, #3B82F6 100%);
  box-shadow: 0 10px 28px rgba(30, 64, 175, .22);
}
.hero .kicker {
  font-size: .76rem; font-weight: 600; letter-spacing: .09em; text-transform: uppercase;
  opacity: .85;
}
.hero .headline { font-size: 1.85rem; font-weight: 700; line-height: 1.22; margin: .35rem 0 .55rem; }
.hero .body { font-size: 1.02rem; line-height: 1.55; opacity: .93; max-width: 58rem; margin: 0; }
.hero .stats { display: flex; flex-wrap: wrap; gap: 14px 36px; margin-top: 20px; }
.hero .stat b { display: block; font-size: 1.45rem; font-weight: 700; font-variant-numeric: tabular-nums; }
.hero .stat span { font-size: .82rem; opacity: .85; }
.chip {
  display: inline-block; padding: 2px 10px; border-radius: 999px; font-size: .8rem;
  font-weight: 600; margin: 0 6px 4px 0; white-space: nowrap;
}
.step {
  display: inline-flex; align-items: center; justify-content: center; width: 26px; height: 26px;
  border-radius: 50%; background: #2563EB; color: #fff; font-weight: 700; font-size: .85rem;
  margin-right: 8px;
}
.question { font-size: .8rem; font-weight: 600; letter-spacing: .06em; text-transform: uppercase; opacity: .7; }
.answer { font-size: 1.2rem; font-weight: 650; line-height: 1.35; margin: .25rem 0 .4rem; }
.card-title { font-weight: 600; font-size: 1.05rem; line-height: 1.35; margin: .2rem 0 .3rem; }
@media (prefers-reduced-motion: no-preference) {
  .hero { animation: rise .4s ease-out both; }
  @keyframes rise { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: none; } }
}
</style>
"""


def clients() -> dict[str, str]:
    found = {}
    for path in sorted(DEPLOY.glob("*/summary.json")):
        found[path.parent.name] = json.loads(path.read_text())["display_name"]
    return found


def pick_client() -> str:
    options = clients()
    if not options:
        st.error("No client outputs in outputs/deploy/. Run `make report CLIENT=...` first.")
        st.stop()
    return st.sidebar.radio("Store chain", list(options), format_func=options.get, key="client")


def current_client() -> str:
    return st.session_state.get("client") or next(iter(clients()))


@st.cache_data
def table(client: str, name: str) -> pl.DataFrame:
    return pl.read_csv(
        DEPLOY / client / name,
        schema_overrides={"product_id": pl.String, "other_product_id": pl.String},
    )


@st.cache_data
def document(client: str, name: str) -> dict:
    return json.loads((DEPLOY / client / name).read_text())


@st.cache_resource
def simulator(client: str) -> Simulator:
    return Simulator.from_dict(document(client, "simulator.json"))


# Scanner names are cut to 20-odd characters. Spell out the ones a shopper wouldn't recognise.
READABLE = {
    "PL HONEY NUT TOASTD OATS": "Store brand Honey Nut Toasted Oats",
    "PL RAISIN BRAN": "Store brand Raisin Bran",
    "PL BT SZ FRSTD SHRD WHT": "Store brand Frosted Shredded Wheat",
    "QKER LIFE ORIGINAL": "Quaker Life",
    "QKER CAP N CRUNCH BERRIES": "Cap'n Crunch's Crunch Berries",
    "QKER CAP N CRUNCH": "Cap'n Crunch",
    "KELL BITE SIZE MINI WHEAT": "Frosted Mini-Wheats Bite Size",
    "KELL FROSTED FLAKES": "Frosted Flakes",
    "KELL FROOT LOOPS": "Froot Loops",
    "GM HONEY NUT CHEERIOS": "Honey Nut Cheerios",
    "GM CHEERIOS": "Cheerios",
    "POST HNY BN OTS HNY RSTD": "Honey Bunches of Oats Honey Roasted",
    "POST FM SZ HNYBNCH OT ALM": "Honey Bunches of Oats Almonds, family size",
    "CAPN CRUNCH CEREAL": "Cap'n Crunch",
    "KELLOGG'S RAISIN BRA": "Kellogg's Raisin Bran",
    "KELLOGG'S FROSTED FL": "Frosted Flakes",
    "KELLOGGS FRUIT LOOPS": "Froot Loops",
    "BITE SIZE FRSTD MINI": "Frosted Mini-Wheats Bite Size",
    "KELL FROST MINI WHTS": "Frosted Mini-Wheats",
    "KELLOGGS RICE KRISPI": "Rice Krispies",
    "DOM FROSTED FLAKES": "Store brand Frosted Flakes",
    "DOM RAISIN BRAN": "Store brand Raisin Bran",
    "DOM CORN FLAKES": "Store brand Corn Flakes",
    "POST GRAPE-NUTS CERE": "Grape-Nuts",
}
MAKERS = {
    "Private Label": "Store brand",
    "Dominicks": "Store brand",
    "General Mi": "General Mills",
    "Kellogg": "Kellogg's",
    "Post Foods": "Post",
}


def label(name: str, size: float) -> str:
    if name in READABLE:
        return f"{READABLE[name]} {size:g} oz"
    # Raw names are upper case with two-letter abbreviations (BT, SZ) that title() would mangle.
    words = [w if len(w) <= 2 and w != "N" else w[0] + w[1:].lower() for w in name.split()]
    return f"{' '.join(words)} {size:g} oz"


def with_label(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.with_columns(
        product=pl.struct("name", "size_oz").map_elements(
            lambda r: label(r["name"], r["size_oz"]), return_dtype=pl.String
        )
    )


def maker(name: str | None) -> str:
    name = (name or "Unknown").title()
    return MAKERS.get(name, name)


@st.cache_data
def maker_colors(client: str) -> dict[str, str]:
    names = sorted({maker(m) for m in table(client, "product_clusters.csv")["manufacturer"]})
    national = [n for n in names if n != "Store brand"]
    colors = dict(zip(national, MAKER_PALETTE * 3, strict=False))
    colors["Store brand"] = SLATE
    return colors


def chip(text: str, color: str) -> str:
    return (
        f'<span class="chip" style="background:{color}22;color:{color};'
        f'border:1px solid {color}55">{html.escape(text)}</span>'
    )


def hero(kicker: str, headline: str, body: str, stats: list[tuple[str, str]] = ()) -> None:
    cells = "".join(f'<div class="stat"><b>{v}</b><span>{k}</span></div>' for k, v in stats)
    st.html(
        f'<div class="hero"><div class="kicker">{kicker}</div>'
        f'<div class="headline">{headline}</div><p class="body">{body}</p>'
        f"{f'<div class=stats>{cells}</div>' if stats else ''}</div>"
    )


def money(value: float, signed: bool = False) -> str:
    """Dollar amount for markdown text: the $ is escaped so Streamlit doesn't read it as LaTeX."""
    sign = ("+" if value >= 0 else "-") if signed else ("-" if value < 0 else "")
    return f"{sign}\\${abs(value):,.0f}"


@st.cache_data
def findings(client: str) -> dict:
    """The headline numbers the briefing and comparison pages are written from."""
    summary = document(client, "summary.json")
    weekly = table(client, "weekly_category.csv")
    opt = document(client, "optimizer.json")["margin"]
    prices = pl.DataFrame(opt["prices"]).with_columns(
        change=pl.col("recommended") / pl.col("current") - 1
    )
    m3 = with_label(table(client, "elasticities.csv").filter(pl.col("model") == "M3 hierarchical"))
    backtest = table(client, "backtest_summary.csv").filter(pl.col("segment") == "all")
    plan = table(client, "promo_plan.csv").filter(pl.col("note").is_null() | (pl.col("note") == ""))
    mechanics = (
        plan.group_by("scenario")
        .agg(
            margin=pl.col("category_incremental_margin").mean(),
            losing=(pl.col("category_incremental_margin") < 0).mean(),
        )
        .sort("margin", descending=True)
    )
    by_tier = (
        m3.join(
            table(client, "product_clusters.csv").select("product_id", "is_private_label"),
            on="product_id",
        )
        .group_by("is_private_label")
        .agg(pl.col("own_elasticity").median())
    )
    tiers = {bool(r[0]): r[1] for r in by_tier.iter_rows()}
    wape = dict(backtest.select("model", "wape_chain_week").iter_rows())
    return {
        "name": summary["display_name"],
        "short_name": summary["display_name"].split(" (")[0],
        "stores": summary["stores"],
        "products": summary["products"],
        "weekly_revenue": weekly.tail(52)["revenue"].mean(),
        "promo_unit_share": weekly.tail(52)["promo_unit_share"].mean(),
        "gain": opt["weekly_gain"],
        "gain_pct": opt["gain_pct"],
        "band": opt["error_band_weekly"],
        "margin_today": opt["current_weekly_value"],
        "units_change_pct": opt["optimized_weekly_units"] / opt["current_weekly_units"] - 1,
        "raised": int((prices["change"] > 0.001).sum()),
        "cut": int((prices["change"] < -0.001).sum()),
        "held": int((prices["change"].abs() <= 0.001).sum()),
        "biggest_rise": prices.sort("change").row(-1, named=True),
        "median_elasticity": m3["own_elasticity"].median(),
        # Most and least sensitive among estimates the data supports, not the noisy extremes.
        "most_sensitive": m3.filter("plausible").sort("own_elasticity").row(0, named=True),
        "least_sensitive": m3.filter("plausible").sort("own_elasticity").row(-1, named=True),
        "national_elasticity": tiers.get(False),
        "private_label_elasticity": tiers.get(True),
        "promo_ratio": (m3["promo_elasticity"] / m3["own_elasticity"]).median(),
        "best_mechanic": mechanics.row(0, named=True),
        "mechanics": mechanics,
        "wape_m3": wape.get("M3 hierarchical"),
        "wape_naive": wape.get("M0 naive"),
        "wape_models": wape,
        "has_feature": summary["has_feature"],
        "has_display": summary["has_display"],
    }


@st.cache_data
def price_curve(client: str, product_id: str, reach: float = 0.2, steps: int = 41) -> pl.DataFrame:
    """Weekly units and margin as one product's regular price moves, everything else fixed."""
    sim = simulator(client)
    rows = []
    for change in np.linspace(-reach, reach, steps):
        t = sim.run(Scenario({product_id: float(change)}, {}, {}, {}))["products"]
        me = t.filter(pl.col("product_id") == product_id).row(0, named=True)
        rows.append(
            {
                "change": float(change),
                "price": me["scenario_price"],
                "units": me["scenario_units"],
                "margin": me["scenario_margin"],
                "category_units": float(t["scenario_units"].sum()),
                "category_margin": float(t["scenario_margin"].sum()),
            }
        )
    return pl.DataFrame(rows)


def header(title: str, lede: str) -> None:
    st.html(f'<h1 class="page-title">{title}</h1><p class="lede">{lede}</p>')


def glossary(*terms: str) -> None:
    with st.expander("What these terms mean", icon=":material/menu_book:"):
        for term in terms:
            st.markdown(f"**{term}**: {GLOSSARY[term]}")


def plot(fig: go.Figure, height: int, legend: bool = True) -> None:
    fig.update_layout(
        height=height,
        font_family="IBM Plex Sans, sans-serif",
        margin={"l": 8, "r": 8, "t": 40 if legend else 16, "b": 8},
        showlegend=legend,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0, "title_text": ""},
        hoverlabel={"font_family": "IBM Plex Sans, sans-serif"},
    )
    st.plotly_chart(
        fig,
        config={
            "displaylogo": False,
            "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"],
        },
    )
