import plotly.express as px
import plotly.graph_objects as go
import polars as pl
import streamlit as st
from shared import label, pick_client, table

client = pick_client()
st.title("Price sensitivity")
e = table(client, "elasticities.csv")
models = [
    m
    for m in e["model"].unique(maintain_order=True).to_list()
    if e.filter(pl.col("model") == m)["own_elasticity"].drop_nulls().len()
]
model = st.selectbox(
    "Model", models, index=models.index("M3 hierarchical") if "M3 hierarchical" in models else 0
)
st.caption(
    "Own-price elasticity of the regular price: % change in weekly units for a 1% permanent price rise. Bars outside the plausible range are orange."
)

data = (
    e.filter(pl.col("model") == model)
    .with_columns(
        product=pl.struct("name", "size_oz").map_elements(
            lambda r: label(r["name"], r["size_oz"]), return_dtype=pl.String
        )
    )
    .sort("own_elasticity")
)
error = None
if data["own_lower"].drop_nulls().len():
    error = {
        "type": "data",
        "symmetric": False,
        "array": (data["own_upper"] - data["own_elasticity"]).to_list(),
        "arrayminus": (data["own_elasticity"] - data["own_lower"]).to_list(),
    }
fig = go.Figure(
    go.Bar(
        x=data["own_elasticity"],
        y=data["product"],
        orientation="h",
        error_x=error,
        marker_color=["#2f5d8a" if p else "#c2662d" for p in data["plausible"]],
    )
)
fig.update_layout(
    height=24 * data.height + 120,
    xaxis_title="Own-price elasticity (95% interval where the model gives one)",
    yaxis_title=None,
    margin={"l": 10},
)
st.plotly_chart(fig, use_container_width=True)

st.subheader("Regular price vs temporary cut")
st.caption(
    "A temporary cut moves units more than the same permanent change. Pricing off promo response overstates how much a permanent cut would sell."
)
compare = data.drop_nulls("promo_elasticity").select(
    pl.col("product").alias("Product"),
    pl.col("own_elasticity").round(2).alias("Regular-price elasticity"),
    pl.col("promo_elasticity").round(2).alias("Temporary-cut elasticity"),
)
st.dataframe(compare, use_container_width=True, hide_index=True)

st.subheader("Cross-price elasticities within clusters (nested logit)")
st.caption(
    "Cell (row, column): % change in the row product's units when the column product's price rises 1%. Positive means they substitute."
)
clusters = table(client, "product_clusters.csv")
names = {r["product_id"]: label(r["name"], r["size_oz"]) for r in clusters.iter_rows(named=True)}
cluster_names = dict(
    clusters.select("cluster", "cluster_name").unique().sort("cluster").iter_rows()
)
cluster = st.selectbox(
    "Product cluster", list(cluster_names), format_func=lambda c: f"{c}: {cluster_names[c]}"
)
members = clusters.filter(pl.col("cluster") == cluster)["product_id"].to_list()
pairs = table(client, "logit_cross_elasticities.csv").filter(
    pl.col("product_id").is_in(members) & pl.col("other_product_id").is_in(members)
)
matrix = pairs.pivot(on="other_product_id", index="product_id", values="nested_logit").sort(
    "product_id"
)
columns = [c for c in matrix.columns if c != "product_id"]
heat = px.imshow(
    matrix.select(columns).to_numpy(),
    x=[names[c] for c in columns],
    y=[names[r] for r in matrix["product_id"]],
    color_continuous_scale="Blues",
    labels={"color": "Cross elasticity"},
    aspect="auto",
    text_auto=".3f",
)
heat.update_layout(height=60 * len(columns) + 200)
st.plotly_chart(heat, use_container_width=True)
