from pathlib import Path

import streamlit as st
from shared import CSS, document, pick_client

STATIC = Path(__file__).parent / "static"

st.set_page_config(page_title="Shelf Simulator", page_icon=str(STATIC / "mark.svg"), layout="wide")
st.html(CSS)
st.logo(str(STATIC / "logo.svg"), icon_image=str(STATIC / "mark.svg"), size="large")

page = st.navigation(
    {
        "Start here": [
            st.Page("views/home.py", title="Summary", icon=":material/home:", default=True),
            st.Page("views/prices.py", title="New price list", icon=":material/sell:"),
            st.Page("views/promos.py", title="Promotions", icon=":material/campaign:"),
            st.Page("views/try_price.py", title="Try a price", icon=":material/tune:"),
        ],
        "For analysts": [
            st.Page("views/simulator.py", title="Scenario builder", icon=":material/table_edit:"),
            st.Page(
                "views/elasticities.py", title="Price sensitivity", icon=":material/trending_down:"
            ),
            st.Page("views/compare.py", title="Compare chains", icon=":material/compare_arrows:"),
        ],
    }
)

client = pick_client()
summary = document(client, "summary.json")
st.sidebar.caption(
    f"{summary['stores']} stores, {summary['products']} cereals, "
    f"{summary['first_week'][:4]} to {summary['last_week'][:4]}. "
    "Built on public sales data; not affiliated with either chain."
)
page.run()
