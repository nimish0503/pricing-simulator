import plotly.graph_objects as go
import polars as pl
import streamlit as st
from shared import (
    AMBER,
    BLUE,
    GREY,
    MODELS,
    PLAUSIBLE,
    current_client,
    glossary,
    hero,
    label,
    plot,
    table,
    with_label,
)

client = current_client()

e = with_label(table(client, "elasticities.csv"))
models = [
    m
    for m in e["model"].unique(maintain_order=True).to_list()
    if e.filter(pl.col("model") == m)["own_elasticity"].drop_nulls().len()
]
model = st.segmented_control(
    "Model",
    models,
    default="M3 hierarchical" if "M3 hierarchical" in models else models[0],
    format_func=lambda m: MODELS.get(m, m),
    required=True,
    help="Four demand models fitted to the same data. Hierarchical Bayes is the one the "
    "recommendations use: it forecasts close to the best and gives the most stable elasticities.",
)
data = e.filter(pl.col("model") == model).drop_nulls("own_elasticity").sort("own_elasticity")

most, least = data.row(0, named=True), data.row(-1, named=True)
supported = data.filter("plausible")
hero(
    f"Price sensitivity · {MODELS.get(model, model)}",
    f"A 1% price rise costs the typical product {-data['own_elasticity'].median():.1f}% of "
    "its units",
    f"From {supported['product'][0]} ({supported['own_elasticity'][0]:.1f}) to "
    f"{supported['product'][-1]} ({supported['own_elasticity'][-1]:.1f}) among the "
    f"{supported.height} products with believable estimates. The less sensitive a product, "
    "the more room there is to raise its price."
    if supported.height
    else "None of this model's estimates fall in the expected range.",
)
cols = st.columns(4)
cols[0].metric(
    "Typical elasticity",
    f"{data['own_elasticity'].median():.2f}",
    help="Median across products. -2 means a 1% price rise loses 2% of units.",
    border=True,
)
cols[1].metric(
    "Most sensitive",
    f"{most['own_elasticity']:.2f}",
    most["product"],
    "off",
    delta_arrow="off",
    border=True,
)
cols[2].metric(
    "Least sensitive",
    f"{least['own_elasticity']:.2f}",
    least["product"],
    "off",
    delta_arrow="off",
    border=True,
)
cols[3].metric(
    "In the expected range",
    f"{data['plausible'].sum()} of {data.height}",
    help=f"Elasticities between {PLAUSIBLE[0]:g} and {PLAUSIBLE[1]:g}.",
    border=True,
)

st.subheader("Regular-price elasticity by product")
fig = go.Figure()
fig.add_vrect(
    x0=PLAUSIBLE[0],
    x1=PLAUSIBLE[1],
    fillcolor=BLUE,
    opacity=0.07,
    line_width=0,
    annotation_text="Expected range",
    annotation_position="top left",
)
has_interval = data["own_lower"].drop_nulls().len() > 0
for plausible, name, color, symbol in [
    (True, "In the expected range", BLUE, "circle"),
    (False, "Outside it: treat with caution", AMBER, "diamond"),
]:
    part = data.filter(pl.col("plausible") == plausible)
    fig.add_trace(
        go.Scatter(
            x=part["own_elasticity"],
            y=part["product"],
            mode="markers",
            name=name,
            marker={"color": color, "size": 11, "symbol": symbol},
            error_x={
                "type": "data",
                "symmetric": False,
                "array": (part["own_upper"] - part["own_elasticity"]).to_list(),
                "arrayminus": (part["own_elasticity"] - part["own_lower"]).to_list(),
                "color": color,
                "thickness": 1.5,
                "width": 0,
            }
            if has_interval
            else None,
            hovertemplate="%{y}<br>Elasticity %{x:.2f}<extra></extra>",
        )
    )
fig.update_yaxes(categoryorder="array", categoryarray=data["product"].to_list()[::-1], title=None)
fig.add_vline(x=0, line_color=GREY, line_width=1)
fig.update_xaxes(title="% change in weekly units for a 1% regular price rise")
plot(fig, 28 * data.height + 140)
if has_interval:
    st.caption(
        "Lines show the 95% interval. A long line means the data can't pin that product down."
    )

st.subheader("Permanent price change vs temporary price cut")
st.caption(
    "Shoppers react more to a short promo than to the same permanent change, because they "
    "stock up on deals. Setting regular prices off promo response would overstate the gain "
    "from a permanent cut."
)
both = data.drop_nulls("promo_elasticity")
if both.height:
    fig = go.Figure()
    for row in both.iter_rows(named=True):
        fig.add_trace(
            go.Scatter(
                x=[row["own_elasticity"], row["promo_elasticity"]],
                y=[row["product"]] * 2,
                mode="lines",
                line={"color": GREY, "width": 2},
                showlegend=False,
                hoverinfo="skip",
            )
        )
    for column, name, color in [
        ("own_elasticity", "Permanent price change", BLUE),
        ("promo_elasticity", "Temporary price cut", AMBER),
    ]:
        fig.add_trace(
            go.Scatter(
                x=both[column],
                y=both["product"],
                mode="markers",
                name=name,
                marker={"color": color, "size": 10},
                hovertemplate=f"%{{y}}<br>{name}: %{{x:.2f}}<extra></extra>",
            )
        )
    fig.update_yaxes(
        categoryorder="array", categoryarray=both["product"].to_list()[::-1], title=None
    )
    fig.update_xaxes(title="% change in weekly units per 1% price change")
    plot(fig, 28 * both.height + 140)
else:
    st.info("This model doesn't estimate a separate promo response.", icon=":material/info:")

st.subheader("Which products steal sales from each other")
st.caption(
    "From the shopper choice model. Read across a row: how much that product's units rise when "
    "the column product's price rises 1%. Bigger numbers mean closer substitutes."
)
clusters = table(client, "product_clusters.csv")
names = {r["product_id"]: label(r["name"], r["size_oz"]) for r in clusters.iter_rows(named=True)}
cluster_names = dict(
    clusters.select("cluster", "cluster_name").unique().sort("cluster").iter_rows()
)
cluster = st.segmented_control(
    "Product cluster",
    list(cluster_names),
    default=next(iter(cluster_names)),
    format_func=lambda c: cluster_names[c].capitalize(),
    required=True,
    help="Products grouped by price tier, pack size and promo behaviour.",
)
members = clusters.filter(pl.col("cluster") == cluster)["product_id"].to_list()
pairs = table(client, "logit_cross_elasticities.csv").filter(
    pl.col("product_id").is_in(members) & pl.col("other_product_id").is_in(members)
)
if pairs.height:
    matrix = pairs.pivot(on="other_product_id", index="product_id", values="nested_logit").sort(
        "product_id"
    )
    columns = sorted(c for c in matrix.columns if c != "product_id")
    fig = go.Figure(
        go.Heatmap(
            z=matrix.select(columns).to_numpy(),
            x=[names[c] for c in columns],
            y=[names[r] for r in matrix["product_id"]],
            colorscale=[[0, "rgba(37,99,235,0.06)"], [1, BLUE]],
            zmin=0,
            texttemplate="%{z:.2f}",
            colorbar={"title": {"text": "Cross<br>elasticity"}, "thickness": 12},
            hovertemplate="%{y} gains %{z:.2f}%<br>when %{x} costs 1% more<extra></extra>",
            xgap=2,
            ygap=2,
        )
    )
    fig.update_yaxes(autorange="reversed", title="Sales of")
    fig.update_xaxes(title="Price rise on", tickangle=-30)
    plot(fig, 56 * len(columns) + 220, legend=False)
else:
    st.info("This cluster has a single product, so there's nothing to switch to inside it.")

glossary(
    "Price elasticity",
    "Expected range",
    "Temporary-cut elasticity",
    "Cross elasticity",
    "Product cluster",
)
