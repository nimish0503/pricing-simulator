"""Loaders shared by the app pages. The app reads only outputs/deploy/, never raw data."""

import json
from pathlib import Path

import polars as pl
import streamlit as st

from shelfsim.choice.simulator import Simulator

DEPLOY = Path(__file__).resolve().parents[1] / "outputs" / "deploy"


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
    return st.sidebar.selectbox("Client", list(options), format_func=options.get, key="client")


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
    return f"{name.title()} {size:g} oz"
