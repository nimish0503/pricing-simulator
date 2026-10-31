import plotly.graph_objects as go
import polars as pl
import streamlit as st
from shared import (
    GREEN,
    PROMOS,
    RED,
    current_client,
    findings,
    header,
    money,
    plot,
    table,
    with_label,
)

client = current_client()
f = findings(client)
header(
    "Do promotions make money?",
    "A promotion here is one week of a deal on one cereal. The profit counts what the deal "
    "adds, minus the profit lost on other cereals shoppers would have bought anyway.",
)

mechanics = f["mechanics"].with_columns(pl.col("scenario").replace(PROMOS)).sort("margin")
st.subheader("Average extra profit from one week of each deal")
fig = go.Figure(
    go.Bar(
        x=mechanics["margin"],
        y=mechanics["scenario"],
        orientation="h",
        marker_color=[GREEN if m > 0 else RED for m in mechanics["margin"]],
        texttemplate="%{x:$,.0f}",
        textposition="outside",
        cliponaxis=False,
        hovertemplate="%{y}: %{x:$,.0f} profit that week<extra></extra>",
    )
)
fig.update_xaxes(
    title="Extra profit that week, all stores (average over cereals)", tickformat="$,.0f"
)
fig.update_yaxes(title=None)
plot(fig, 70 * mechanics.height + 90, legend=False)
st.caption("Green adds profit. Red loses it.")
if not (f["has_feature"] and f["has_display"]):
    st.caption(
        "This chain's data doesn't record ads or in-store displays, so only price cuts are shown."
    )
if (mechanics.filter(pl.col("scenario").is_in(["10% off", "20% off"]))["margin"] < 0).all():
    st.markdown(
        "**Why do price cuts lose money?** Most shoppers who buy the discounted cereal would "
        "have bought a cereal anyway, often the same one at full price. The store gives away "
        "the discount on those boxes and gains few new shoppers."
    )

st.subheader("Best deal for one cereal")
plan = (
    with_label(table(client, "promo_plan.csv"))
    .filter(pl.col("note").is_null() | (pl.col("note") == ""))
    .with_columns(pl.col("scenario").replace(PROMOS))
)
order = (
    table(client, "product_profile.csv")
    .sort("revenue", descending=True)["product_id"]
    .cast(pl.String)
    .to_list()
)
names = dict(zip(plan["product_id"], plan["product"], strict=False))
pid = st.selectbox("Cereal", [p for p in order if p in names], format_func=names.get)
one = plan.filter(pl.col("product_id") == pid).sort("category_incremental_margin", descending=True)
best, worst = one.row(0, named=True), one.row(-1, named=True)
for title, row in [("Best deal", best), ("Worst deal", worst)]:
    value = row["category_incremental_margin"]
    st.markdown(
        f"**{title}:** {row['scenario']}, which {'adds' if value >= 0 else 'loses'} about "
        f"**{money(abs(value))}** profit that week."
    )

with st.expander("Best deal for every cereal", icon=":material/table_rows:"):
    st.dataframe(
        plan.sort("category_incremental_margin", descending=True)
        .group_by("product", maintain_order=True)
        .first()
        .select("product", "scenario", pl.col("category_incremental_margin").round(0)),
        hide_index=True,
        column_config={
            "product": st.column_config.TextColumn("Cereal", width="large"),
            "scenario": "Best deal",
            "category_incremental_margin": st.column_config.NumberColumn(
                "Extra profit that week", format="dollar"
            ),
        },
    )
