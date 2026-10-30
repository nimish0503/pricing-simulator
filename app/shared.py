"""Loaders and page components shared by the app. The app reads only outputs/deploy/, never raw data."""

import json
from pathlib import Path

import plotly.graph_objects as go
import polars as pl
import streamlit as st

from shelfsim.choice.simulator import Simulator
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

CSS = """
<style>
.block-container { padding-top: 2.2rem; max-width: 1280px; }
.page-title { font-size: 2rem; font-weight: 700; line-height: 1.2; margin: 0 0 .5rem; padding: 0; }
.lede { font-size: 1.02rem; opacity: .78; max-width: 52rem; margin: 0 0 .6rem; }
[data-testid="stMetricValue"], [data-testid="stDataFrame"] { font-variant-numeric: tabular-nums; }
[data-testid="stMetricLabel"] p { font-weight: 500; }
h2 { margin-top: .8rem; }
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
    return st.sidebar.radio("Retailer", list(options), format_func=options.get, key="client")


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


def label(name: str, size: float) -> str:
    # Raw names are upper case with two-letter abbreviations (PL, GM, BT) that title() would mangle.
    words = [w if len(w) <= 2 and w != "N" else w.title() for w in name.split()]
    return f"{' '.join(words)} {size:g} oz"


def with_label(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.with_columns(
        product=pl.struct("name", "size_oz").map_elements(
            lambda r: label(r["name"], r["size_oz"]), return_dtype=pl.String
        )
    )


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
