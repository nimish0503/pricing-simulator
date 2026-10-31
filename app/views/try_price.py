import plotly.graph_objects as go
import polars as pl
import streamlit as st
from plotly.subplots import make_subplots
from shared import (
    AMBER,
    BLUE,
    GREEN,
    SETTINGS,
    SLATE,
    current_client,
    document,
    header,
    money,
    plot,
    price_curve,
    simulator,
    table,
    with_label,
)

from shelfsim.choice.simulator import Scenario

client = current_client()
header(
    "Try a price",
    "Pick a cereal and move its price. The tool estimates what happens in a typical week "
    "across all stores, with every other price left as it is today.",
)

products = with_label(table(client, "product_clusters.csv")).with_columns(
    pl.col("product_id").cast(pl.String)
)
names = dict(zip(products["product_id"], products["product"], strict=True))
order = (
    table(client, "product_profile.csv")
    .sort("revenue", descending=True)["product_id"]
    .cast(pl.String)
    .to_list()
)
suggested = {p["product_id"]: p for p in document(client, "optimizer.json")["margin"]["prices"]}

left, right = st.columns([1, 1.4], gap="large")
with left:
    pid = st.selectbox("Cereal", order, format_func=names.get, key=f"try_{client}")
    today = suggested[pid]["current"]
    change = st.select_slider(
        "New price",
        options=list(range(-20, 21)),
        value=0,
        format_func=lambda c: (
            f"${today * (1 + c / 100):.2f} ({c:+d}%)" if c else f"${today:.2f} (today)"
        ),
        key=f"change_{client}_{pid}",
    )
    st.caption(f"Suggested price on the price list: \\${suggested[pid]['recommended']:.2f}")

result = simulator(client).run(Scenario({pid: change / 100}, {}, {}, {}))
t = result["products"]
me = t.filter(pl.col("product_id") == pid).row(0, named=True)
aisle = {
    k: (float(t[f"base_{k}"].sum()), float(t[f"scenario_{k}"].sum())) for k in ("margin", "revenue")
}
boxes = me["scenario_units"] - me["base_units"]

with left:
    m = st.columns(2)
    m[0].metric(
        "Boxes sold a week",
        f"{me['scenario_units']:,.0f}",
        f"{boxes:+,.0f}" if change else None,
        border=True,
    )
    gain = aisle["margin"][1] - aisle["margin"][0]
    m[1].metric(
        "Cereal aisle profit a week",
        f"${aisle['margin'][1]:,.0f}",
        f"{'+' if gain >= 0 else '-'}${abs(gain):,.0f}" if change else None,
        border=True,
    )

with right:
    if not change:
        st.info("Move the slider to see what happens.", icon=":material/touch_app:")
    else:
        flows = result["switching"]["lost_to" if boxes < 0 else "won_from"]
        if boxes < 0:
            st.markdown(
                f"At **\\${me['scenario_price']:.2f}**, about **{-boxes:,.0f} fewer boxes** of "
                f"{names[pid]} sell each week. Those shoppers:"
            )
            parts = [
                ("Buy another cereal from the same company", flows["own"], BLUE),
                ("Switch to a competitor's cereal", flows["competitors"], AMBER),
                ("Skip cereal that week", flows["outside"], SLATE),
            ]
        else:
            st.markdown(
                f"At **\\${me['scenario_price']:.2f}**, about **{boxes:,.0f} more boxes** of "
                f"{names[pid]} sell each week. Those shoppers come from:"
            )
            parts = [
                ("Other cereals from the same company", flows["own"], BLUE),
                ("Competitors' cereals", flows["competitors"], AMBER),
                ("People who wouldn't have bought cereal", flows["outside"], SLATE),
            ]
        fig = go.Figure(
            go.Bar(
                x=[p[1] for p in parts],
                y=[p[0] for p in parts],
                orientation="h",
                marker_color=[p[2] for p in parts],
                texttemplate="%{x:,.0f} boxes",
                textposition="auto",
                hovertemplate="%{y}: %{x:,.0f} boxes a week<extra></extra>",
            )
        )
        fig.update_yaxes(autorange="reversed", title=None)
        fig.update_xaxes(title="Boxes a week, all stores", tickformat=",")
        plot(fig, 220, legend=False)
        st.markdown(
            f"Overall, the cereal aisle {'gains' if gain >= 0 else 'loses'} "
            f"**{money(abs(gain))}** of profit a week."
        )

st.subheader(f"Cereal aisle profit at each price of {names[pid]}")
curve = price_curve(client, pid)
fig = make_subplots()
fig.add_trace(
    go.Scatter(
        x=curve["price"],
        y=curve["category_margin"],
        name="Cereal aisle profit a week",
        line={"color": AMBER, "width": 3},
        hovertemplate="At %{x:$.2f}: %{y:$,.0f} a week<extra></extra>",
    )
)
picked = curve.filter((pl.col("change") * 100).round(0) == change).row(0, named=True)
fig.add_trace(
    go.Scatter(
        x=[picked["price"]],
        y=[picked["category_margin"]],
        mode="markers",
        name="Your price",
        marker={"color": BLUE, "size": 14, "line": {"color": "white", "width": 2}},
        hovertemplate="Your price %{x:$.2f}: %{y:$,.0f}<extra></extra>",
    )
)
for x, text, colour in [
    (today, "Today", SLATE),
    (suggested[pid]["recommended"], "Suggested", GREEN),
]:
    fig.add_vline(x=x, line_color=colour, line_dash="dash", line_width=1.5)
    fig.add_annotation(
        x=x, y=1.06, yref="paper", text=text, showarrow=False, font={"color": colour}
    )
fig.update_xaxes(title=f"Price of {names[pid]}", tickprefix="$", tickformat=".2f")
fig.update_yaxes(title="Profit a week, all stores", tickformat="$,.0f")
plot(fig, 360)
st.caption(
    "The suggested price may not sit at the top of this line. The price list moves every "
    "cereal at once and keeps to store rules: no price moves more than "
    f"{SETTINGS.optimizer.max_price_change:.0%}, and total boxes sold can't drop more than "
    f"{SETTINGS.optimizer.max_category_volume_drop:.0%}."
)

with st.expander("Price and sales history for this cereal", icon=":material/history:"):
    weekly = (
        table(client, "product_weekly.csv")
        .with_columns(pl.col("product_id").cast(pl.String))
        .filter(pl.col("product_id") == pid)
        .with_columns(pl.col("week_start").str.to_date())
        .sort("week_start")
        .upsample("week_start", every="1w")
    )
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08)
    fig.add_trace(
        go.Scatter(
            x=weekly["week_start"],
            y=weekly["shelf_price"],
            name="Price paid",
            line={"color": BLUE, "width": 1.6},
            hovertemplate="%{y:$.2f}<extra>Price paid</extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=weekly["week_start"],
            y=weekly["units"],
            name="Boxes sold",
            marker_color=AMBER,
            hovertemplate="%{y:,.0f} boxes<extra></extra>",
        ),
        row=2,
        col=1,
    )
    fig.update_yaxes(title_text="Price", tickprefix="$", row=1, col=1)
    fig.update_yaxes(title_text="Boxes a week", tickformat="~s", row=2, col=1)
    fig.update_xaxes(title_text="Week", row=2, col=1)
    fig.update_layout(hovermode="x unified")
    plot(fig, 400)
    st.caption("The tall bars are almost always weeks when the price dipped for a deal.")
