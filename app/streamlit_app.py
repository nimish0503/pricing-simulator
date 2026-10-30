from pathlib import Path

import streamlit as st
from shared import CSS, document, pick_client

STATIC = Path(__file__).parent / "static"

st.set_page_config(page_title="Shelf Simulator", page_icon=str(STATIC / "mark.svg"), layout="wide")
st.html(CSS)
st.logo(str(STATIC / "logo.svg"), icon_image=str(STATIC / "mark.svg"), size="large")

page = st.navigation(
    {
        "Understand": [
            st.Page(
                "views/overview.py", title="Briefing", icon=":material/summarize:", default=True
            ),
            st.Page(
                "views/product.py", title="Product deep-dive", icon=":material/search_insights:"
            ),
            st.Page(
                "views/elasticities.py", title="Price sensitivity", icon=":material/trending_down:"
            ),
            st.Page(
                "views/compare.py", title="Compare retailers", icon=":material/compare_arrows:"
            ),
        ],
        "Decide": [
            st.Page("views/simulator.py", title="What-if", icon=":material/tune:"),
            st.Page("views/optimizer.py", title="Recommendations", icon=":material/price_check:"),
        ],
    }
)

client = pick_client()
summary = document(client, "summary.json")
st.sidebar.caption(
    f"{summary['stores']} stores · {summary['products']} products · {summary['weeks']} weeks, "
    f"{summary['first_week'][:4]} to {summary['last_week'][:4]}"
)
st.sidebar.caption(
    "Built on public scanner data (Dominick's Finer Foods, dunnhumby). "
    "Not affiliated with either retailer or any pricing vendor."
)
page.run()
