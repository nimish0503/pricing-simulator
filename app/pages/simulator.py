import time

import plotly.graph_objects as go
import polars as pl
import streamlit as st
from shared import document, label, pick_client, simulator

from shelfsim.choice.simulator import Scenario

client = pick_client()
sim = simulator(client)
summary = document(client, "summary.json")
st.title("Scenario simulator")
st.caption(
    "Weekly effect across all stores of permanent price changes and a one-week promo, from 10,000 virtual shoppers per store cluster."
)

products = sim.products.with_columns(
    product=pl.struct("name", "size_oz").map_elements(
        lambda r: label(r["name"], r["size_oz"]), return_dtype=pl.String
    )
)
by_label = dict(zip(products["product"], products["product_id"], strict=True))
chosen = st.multiselect("Products to change", list(by_label), default=list(by_label)[:1])

changes, cuts, features, displays = {}, {}, {}, {}
for name in chosen:
    pid = by_label[name]
    st.markdown(f"**{name}**")
    c1, c2, c3, c4 = st.columns([3, 3, 1, 1])
    changes[pid] = (
        c1.slider("Permanent price change", -10, 10, 0, format="%d%%", key=f"p{pid}") / 100
    )
    cuts[pid] = (
        c2.slider("Temporary price cut", 0, 30, 0, step=5, format="%d%%", key=f"c{pid}") / 100
    )
    features[pid] = c3.checkbox("Feature", key=f"f{pid}", disabled=not summary["has_feature"])
    displays[pid] = c4.checkbox("Display", key=f"d{pid}", disabled=not summary["has_display"])

started = time.perf_counter()
result = sim.run(Scenario(changes, cuts, features, displays))
elapsed = time.perf_counter() - started
t = result["products"]


def delta(column):
    return float(t[f"scenario_{column}"].sum() - t[f"base_{column}"].sum())


cols = st.columns(3)
cols[0].metric(
    "Category units / week", f"{t['scenario_units'].sum():,.0f}", f"{delta('units'):+,.0f}"
)
cols[1].metric(
    "Category revenue / week", f"${t['scenario_revenue'].sum():,.0f}", f"{delta('revenue'):+,.0f}"
)
cols[2].metric(
    "Category gross margin / week",
    f"${t['scenario_margin'].sum():,.0f}",
    f"{delta('margin'):+,.0f}",
)
st.caption(f"Simulated in {elapsed:.2f} s.")

flows = result["switching"]
if chosen:
    focal = t.filter(pl.col("product_id").is_in([by_label[c] for c in chosen]))
    change = float((focal["scenario_units"] - focal["base_units"]).sum())
    st.subheader("Where the volume went")
    if change < 0:
        steps = [
            ("Lost to own brands", -flows["lost_to"]["own"]),
            ("Lost to competitors", -flows["lost_to"]["competitors"]),
            ("Stopped buying cereal", -flows["lost_to"]["outside"]),
        ]
    else:
        steps = [
            ("Won from own brands", flows["won_from"]["own"]),
            ("Won from competitors", flows["won_from"]["competitors"]),
            ("New category buyers", flows["won_from"]["outside"]),
        ]
    fig = go.Figure(
        go.Waterfall(
            x=[s[0] for s in steps] + ["Net change, changed products"],
            y=[s[1] for s in steps] + [None],
            measure=["relative"] * len(steps) + ["total"],
            connector={"line": {"color": "#999"}},
        )
    )
    fig.update_layout(yaxis_title="Units per week, all stores", height=380)
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Switching is counted from simulated shoppers who change their pick, so it carries some simulation noise relative to the unit totals above."
    )

st.subheader("By product")
view = t.with_columns(
    units_change=pl.col("scenario_units") - pl.col("base_units"),
    revenue_change=pl.col("scenario_revenue") - pl.col("base_revenue"),
).sort("units_change")
st.dataframe(
    view.select(
        pl.col("name").alias("Product"),
        pl.col("scenario_price").round(2).alias("Price paid ($)"),
        pl.col("base_units").round(0).alias("Units / week, today"),
        pl.col("units_change").round(0).alias("Change in units"),
        pl.col("revenue_change").round(0).alias("Change in revenue ($)"),
    ),
    use_container_width=True,
    hide_index=True,
)
