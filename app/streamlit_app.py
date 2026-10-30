import streamlit as st

st.set_page_config(page_title="Shelf Simulator", layout="wide")
st.navigation(
    [
        st.Page("pages/overview.py", title="Client overview", default=True),
        st.Page("pages/elasticities.py", title="Elasticities"),
        st.Page("pages/simulator.py", title="Scenario simulator"),
        st.Page("pages/optimizer.py", title="Optimizer"),
    ]
).run()
