import plotly.graph_objects as go
import polars as pl
import streamlit as st
from plotly.subplots import make_subplots
from shared import (
    AMBER,
    BLUE,
    GREEN,
    GREY,
    RED,
    SETTINGS,
    SLATE,
    chip,
    current_client,
    document,
    glossary,
    hero,
    maker,
    maker_colors,
    plot,
    price_curve,
    simulator,
    table,
    with_label,
)

from shelfsim.choice.simulator import Scenario

client = current_client()
products = with_label(table(client, "product_clusters.csv"))
profile = table(client, "product_profile.csv").sort("revenue", descending=True)
ids = profile["product_id"].to_list()
names = dict(zip(products["product_id"], products["product"], strict=True))
wanted = st.session_state.get("product")
pid = st.selectbox(
    "Product",
    ids,
    index=ids.index(wanted) if wanted in ids else 0,
    format_func=names.get,
    help="Ordered by revenue. You can also click a product on the briefing page.",
)
st.session_state["product"] = pid

info = products.filter(pl.col("product_id") == pid).row(0, named=True)
prof = profile.filter(pl.col("product_id") == pid).row(0, named=True)
m3 = table(client, "elasticities.csv").filter(
    (pl.col("model") == "M3 hierarchical") & (pl.col("product_id") == pid)
)
own = m3["own_elasticity"][0] if m3.height else None
promo = m3["promo_elasticity"][0] if m3.height else None
price = next(
    p for p in document(client, "optimizer.json")["margin"]["prices"] if p["product_id"] == pid
)
change = price["recommended"] / price["current"] - 1

color = maker_colors(client).get(maker(info["manufacturer"]), SLATE)
st.html(
    chip(maker(info["manufacturer"]), color)
    + chip(info["cluster_name"].capitalize(), SLATE)
    + chip(f"{info['size_oz']:g} oz pack", SLATE)
)
action = (
    f"Raise to ${price['recommended']:.2f} ({change:+.1%})"
    if change > 0.001
    else f"Cut to ${price['recommended']:.2f} ({change:+.1%})"
    if change < -0.001
    else f"Hold at ${price['current']:.2f}"
)
sensitivity = (
    f"A 1% regular price rise loses about {-own:.1f}% of its weekly units"
    if own is not None
    else "No elasticity estimate"
)
hero(
    names[pid],
    action,
    f"{sensitivity}. It sells on a price cut in {prof['price_cut_share']:.0%} of store-weeks, "
    f"{prof['mean_depth_on_cut']:.0%} off on average.",
    [
        ("Regular price today", f"${price['current']:.2f}"),
        ("Recommended", f"${price['recommended']:.2f}"),
        ("Price sensitivity", f"{own:.2f}" if own is not None else "n/a"),
        ("Promo sensitivity", f"{promo:.2f}" if promo is not None else "n/a"),
        ("Revenue, whole period", f"${prof['revenue'] / 1e6:.2f}M"),
    ],
)

st.subheader("Price and sales history")
weekly = (
    table(client, "product_weekly.csv")
    .filter(pl.col("product_id") == pid)
    .with_columns(pl.col("week_start").str.to_date())
    .sort("week_start")
    .upsample("week_start", every="1w")
)
fig = make_subplots(
    rows=2, cols=1, shared_xaxes=True, row_heights=[0.5, 0.5], vertical_spacing=0.07
)
fig.add_trace(
    go.Scatter(
        x=weekly["week_start"],
        y=weekly["regular_price"],
        name="Regular price",
        line={"color": SLATE, "width": 2, "shape": "hv", "dash": "dot"},
        hovertemplate="%{y:$.2f}<extra>Regular price</extra>",
    ),
    row=1,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=weekly["week_start"],
        y=weekly["shelf_price"],
        name="Typical shelf price paid",
        line={"color": BLUE, "width": 1.6},
        hovertemplate="%{y:$.2f}<extra>Shelf price</extra>",
    ),
    row=1,
    col=1,
)
fig.add_trace(
    go.Bar(
        x=weekly["week_start"],
        y=weekly["units"],
        name="Units sold (color: share of stores on a price cut)",
        marker={
            "color": weekly["stores_on_price_cut"].fill_null(0),
            "colorscale": [[0, "rgba(37,99,235,.55)"], [1, AMBER]],
            "cmin": 0,
            "cmax": 1,
        },
        customdata=weekly["stores_on_price_cut"],
        hovertemplate="%{y:,.0f} units<br>%{customdata:.0%} of stores on a cut<extra></extra>",
    ),
    row=2,
    col=1,
)
fig.update_yaxes(title_text="Price ($)", tickprefix="$", row=1, col=1)
fig.update_yaxes(title_text="Units per week", tickformat="~s", row=2, col=1)
fig.update_xaxes(title_text="Week starting", row=2, col=1)
fig.update_layout(hovermode="x unified", bargap=0.1)
plot(fig, 440)
st.caption(
    "Amber bars are weeks when most stores had it on a price cut. The spikes in units "
    "line up with them."
)

left, right = st.columns([1.25, 1], gap="large")
with left:
    st.subheader("What happens at other prices")
    curve = price_curve(client, pid)
    limit = SETTINGS.optimizer.max_price_change
    best = (
        curve.filter(pl.col("change").abs() <= limit + 1e-9)
        .sort("category_margin")
        .row(-1, named=True)
    )
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_vrect(
        x0=price["current"] * (1 - limit),
        x1=price["current"] * (1 + limit),
        fillcolor=BLUE,
        opacity=0.06,
        line_width=0,
        annotation_text=f"Allowed range (±{limit:.0%})",
        annotation_position="top left",
    )
    fig.add_trace(
        go.Scatter(
            x=curve["price"],
            y=curve["units"],
            name="This product's units per week",
            line={"color": BLUE, "width": 2.5},
            hovertemplate="At %{x:$.2f}: %{y:,.0f} units<extra></extra>",
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=curve["price"],
            y=curve["category_margin"],
            name="Category margin per week",
            line={"color": AMBER, "width": 2.5},
            hovertemplate="At %{x:$.2f}: %{y:$,.0f} category margin<extra></extra>",
        ),
        secondary_y=True,
    )
    for x, text, colour, side in [
        (price["current"], f"Today {price['current']:.2f}", SLATE, "right"),
        (price["recommended"], f"Recommended {price['recommended']:.2f}", GREEN, "left"),
    ]:
        fig.add_vline(x=x, line_color=colour, line_dash="dash", line_width=1.5)
        fig.add_annotation(
            x=x,
            y=1.04,
            yref="paper",
            text=text,
            showarrow=False,
            xanchor=side,
            font={"color": colour, "size": 12},
        )
    fig.add_trace(
        go.Scatter(
            x=[best["price"]],
            y=[best["category_margin"]],
            mode="markers",
            name="Best category margin for this product alone",
            marker={"color": AMBER, "size": 11, "symbol": "star"},
            hovertemplate="%{x:$.2f}: %{y:$,.0f}<extra>Best in range</extra>",
        ),
        secondary_y=True,
    )
    fig.update_xaxes(title="Regular price", tickprefix="$")
    fig.update_yaxes(title="Units per week", tickformat=",", secondary_y=False)
    fig.update_yaxes(
        title="Category margin per week", tickformat="$,.0f", secondary_y=True, showgrid=False
    )
    fig.update_layout(legend={"y": -0.32, "yanchor": "top"})
    plot(fig, 380)
    st.caption(
        "From the shopper simulator, moving only this product's price. The recommended price "
        "can differ from the star because the optimizer moves every price at once."
    )

with right:
    st.subheader(f"Where its shoppers go at +{limit:.0%}")
    sim = simulator(client)
    moved = sim.run(Scenario({pid: limit}, {}, {}, {}))
    flows = moved["switching"]["lost_to"]
    gainers = (
        moved["products"]
        .filter(pl.col("product_id") != pid)
        .with_columns(
            gain=pl.col("scenario_units") - pl.col("base_units"),
            product=pl.col("product_id").replace_strict(names, default=None),
            same=pl.col("product_id").is_in(
                products.filter(pl.col("cluster") == info["cluster"])["product_id"].to_list()
            ),
        )
        .sort("gain", descending=True)
        .head(6)
    )
    bars = gainers.select("product", "gain", "same").vstack(
        pl.DataFrame(
            {
                "product": ["Stop buying in the category"],
                "gain": [flows["outside"]],
                "same": [None],
            },
            schema=gainers.select("product", "gain", "same").schema,
        )
    )
    fig = go.Figure(
        go.Bar(
            x=bars["gain"],
            y=bars["product"],
            orientation="h",
            marker_color=[GREY if s is None else BLUE if s else AMBER for s in bars["same"]],
            texttemplate="+%{x:,.0f}",
            textposition="auto",
            hovertemplate="%{y}: +%{x:,.0f} units a week<extra></extra>",
        )
    )
    fig.update_yaxes(autorange="reversed", title=None)
    fig.update_xaxes(title=f"Extra units a week if this costs {limit:.0%} more", tickformat=",")
    plot(fig, 380, legend=False)
    st.caption(
        "Blue: same cluster. Amber: other clusters. Grey: shoppers who buy no cereal that week. "
        "From the shopper simulator."
    )

st.subheader("Which promo pays for this product")
plan = (
    table(client, "promo_plan.csv")
    .filter(pl.col("product_id") == pid)
    .with_columns(
        scenario=pl.col("scenario").str.slice(0, 1).str.to_uppercase()
        + pl.col("scenario").str.slice(1)
    )
    .sort("category_incremental_margin")
)
p1, p2 = st.columns([1, 1], gap="large")
with p1:
    fig = go.Figure()
    for column, name, colour in [
        ("incremental_margin", "This product's extra margin", BLUE),
        ("category_incremental_margin", "Category's extra margin", AMBER),
    ]:
        fig.add_trace(
            go.Bar(
                x=plan[column],
                y=plan["scenario"],
                orientation="h",
                name=name,
                marker_color=colour,
                hovertemplate=f"%{{y}}: %{{x:$,.0f}}<extra>{name}</extra>",
            )
        )
    fig.update_layout(barmode="group")
    fig.update_xaxes(title="Extra margin in the promo week, all stores", tickformat="$,.0f")
    plot(fig, 320)
with p2:
    best_promo = plan.filter(pl.col("note").is_null() | (pl.col("note") == "")).row(-1, named=True)
    gap = best_promo["incremental_margin"] - best_promo["category_incremental_margin"]
    tone = GREEN if best_promo["category_incremental_margin"] > 0 else RED
    st.html(
        f'<div class="card-title">Best option: <span style="color:{tone}">'
        f"{best_promo['scenario']}</span></div>"
    )
    st.markdown(
        f"It adds **{best_promo['category_incremental_units']:,.0f} category units** in the "
        f"promo week. Of the product's own extra units, "
        f"**{best_promo['own_cannibalization_units']:,.0f}** come from the same maker's other "
        f"products and **{best_promo['competitor_loss_units']:,.0f}** from competitors."
    )
    if gap > 0:
        st.markdown(
            f"The product's own margin looks \\${gap:,.0f} better than the category's, because "
            "part of its gain is taken from other products on the same shelf."
        )
    if (plan["note"].fill_null("") != "").any():
        st.caption("Ads and displays can't be measured for this retailer.")

glossary("Price elasticity", "Temporary-cut elasticity", "Cross elasticity", "Gross margin")
