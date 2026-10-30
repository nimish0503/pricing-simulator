import plotly.express as px
import polars as pl
import streamlit as st
from shared import document, pick_client, table

client = pick_client()
summary = document(client, "summary.json")
st.title(summary["display_name"])
st.caption(
    f"Weeks starting {summary['first_week']} to {summary['last_week']}. "
    "Built from public scanner data; my own take on the problem, not affiliated with any pricing vendor."
)

cols = st.columns(5)
cols[0].metric("Stores", summary["stores"])
cols[1].metric("Products modelled", summary["products"])
cols[2].metric("Weeks with data", summary["weeks"])
cols[3].metric("Zero-sales store-weeks", f"{summary['zero_sales_share']:.1%}")
cols[4].metric("Outlier weeks flagged", f"{summary['outlier_share']:.1%}")
if not (summary["has_feature"] and summary["has_display"]):
    st.info(
        "This client reports no feature or display flags, so those mechanics can't be measured."
    )

weekly = table(client, "weekly_category.csv").with_columns(pl.col("week_start").str.to_date())
st.subheader("Category sales per week, all stores")
fig = px.line(
    weekly,
    x="week_start",
    y="revenue",
    labels={"week_start": "Week starting", "revenue": "Category revenue ($ per week)"},
)
st.plotly_chart(fig, use_container_width=True)
fig = px.line(
    weekly,
    x="week_start",
    y="promo_unit_share",
    labels={
        "week_start": "Week starting",
        "promo_unit_share": "Share of units sold on a price cut",
    },
)
fig.update_yaxes(tickformat=".0%")
st.plotly_chart(fig, use_container_width=True)

st.subheader("Products")
profile = table(client, "product_profile.csv").select(
    pl.col("name").alias("Product"),
    pl.col("manufacturer").alias("Manufacturer"),
    pl.col("size_oz").alias("Size (oz)"),
    pl.col("revenue").round(0).alias("Revenue ($, whole period)"),
    pl.col("price_median").round(2).alias("Median price ($)"),
    pl.col("price_cut_share").round(3).alias("Share of weeks on a price cut"),
    pl.col("mean_depth_on_cut").round(3).alias("Mean cut depth"),
)
st.dataframe(profile, use_container_width=True, hide_index=True)
