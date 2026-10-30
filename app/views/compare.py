import plotly.graph_objects as go
import polars as pl
import streamlit as st
from shared import (
    AMBER,
    BLUE,
    MODELS,
    clients,
    findings,
    header,
    hero,
    plot,
    table,
    with_label,
)

retailers = clients()
COLORS = dict(zip(retailers, [BLUE, AMBER, "#0F766E", "#7C3AED"], strict=False))
f = {c: findings(c) for c in retailers}

header(
    "Two retailers, one method",
    "The same pipeline run on both retailers' cereal data. What differs tells you how much "
    "the answers depend on the shopper base and the promo strategy.",
)
if len(retailers) < 2:
    st.info("Only one retailer has outputs. Run the pipeline for a second one to compare.")
    st.stop()

a, b = sorted(retailers, key=lambda c: f[c]["median_elasticity"])
fa, fb = f[a], f[b]
hero(
    "The short version",
    f"{fa['short_name']} shoppers are more price-sensitive; "
    f"{'its' if fa['wape_m3'] > fb['wape_m3'] else fb['short_name'] + chr(39) + 's'} "
    "sales are harder to forecast",
    f"The typical product loses {-fa['median_elasticity']:.1f}% of units per 1% price rise at "
    f"{fa['short_name']}, against {-fb['median_elasticity']:.1f}% at {fb['short_name']}. "
    f"Weekly category forecasts miss by {fa['wape_m3']:.0%} and {fb['wape_m3']:.0%}."
    + (
        " In both, the optimizer's margin gain is smaller than the forecast error, so the "
        "price changes need a store trial."
        if all(x["band"] > abs(x["gain"]) for x in f.values())
        else ""
    ),
)

cols = st.columns(len(retailers))
for col, client in zip(cols, retailers, strict=True):
    x = f[client]
    with col.container(border=True):
        st.html(f'<div class="card-title" style="color:{COLORS[client]}">{x["name"]}</div>')
        m = st.columns(2)
        m[0].metric("Revenue per week", f"${x['weekly_revenue']:,.0f}")
        m[1].metric("Stores · products", f"{x['stores']} · {x['products']}")
        m = st.columns(2)
        m[0].metric("Units sold on a price cut", f"{x['promo_unit_share']:.0%}")
        m[1].metric(
            "Typical price sensitivity",
            f"{x['median_elasticity']:.2f}",
            help="Median elasticity across products, hierarchical Bayes model.",
        )
        m = st.columns(2)
        m[0].metric(
            "Forecast error",
            f"{x['wape_m3']:.0%}",
            help="Weekly category units, held-out year, hierarchical Bayes model.",
        )
        m[1].metric(
            "Optimizer margin gain",
            f"{x['gain_pct']:+.1%}",
            f"±${x['band']:,.0f} error band",
            "off",
            delta_arrow="off",
        )

st.subheader("How price-sensitive is each product?")
fig = go.Figure()
for client in retailers:
    e = with_label(table(client, "elasticities.csv")).filter(
        (pl.col("model") == "M3 hierarchical") & pl.col("plausible")
    )
    fig.add_trace(
        go.Box(
            x=e["own_elasticity"],
            name=f[client]["short_name"],
            marker_color=COLORS[client],
            boxpoints="all",
            jitter=0.5,
            pointpos=0,
            text=e["product"],
            hovertemplate="%{text}: %{x:.2f}<extra></extra>",
            orientation="h",
        )
    )
fig.update_xaxes(
    title="% change in weekly units for a 1% regular price rise (each dot is a product)"
)
plot(fig, 300, legend=False)
st.caption("Products with estimates outside the expected range are left out.")

left, right = st.columns(2, gap="large")
with left:
    st.subheader("Which model forecasts best?")
    fig = go.Figure()
    for client in retailers:
        wape = f[client]["wape_models"]
        models = [m for m in MODELS if m in wape] + ["M0 naive"]
        fig.add_trace(
            go.Bar(
                y=[MODELS.get(m, "Flat recent average") for m in models],
                x=[wape[m] for m in models],
                name=f[client]["short_name"],
                orientation="h",
                marker_color=COLORS[client],
                hovertemplate="%{y}: %{x:.0%}<extra>" + f[client]["short_name"] + "</extra>",
            )
        )
    fig.update_layout(barmode="group")
    fig.update_xaxes(title="Error on weekly category units (lower is better)", tickformat=".0%")
    fig.update_yaxes(autorange="reversed")
    plot(fig, 380)
with right:
    st.subheader("Do promos pay?")
    fig = go.Figure()
    for client in retailers:
        mech = f[client]["mechanics"]
        fig.add_trace(
            go.Bar(
                y=mech["scenario"].str.slice(0, 1).str.to_uppercase()
                + mech["scenario"].str.slice(1),
                x=mech["margin"],
                name=f[client]["short_name"],
                orientation="h",
                marker_color=COLORS[client],
                hovertemplate="%{y}: %{x:$,.0f}<extra>" + f[client]["short_name"] + "</extra>",
            )
        )
    fig.update_layout(barmode="group")
    fig.update_xaxes(title="Average extra category margin per promo week", tickformat="$,.0f")
    plot(fig, 380)
    st.caption("Mechanics a retailer's data can't measure (ads, displays) are left out.")
