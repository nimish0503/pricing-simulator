import plotly.graph_objects as go
import polars as pl
import streamlit as st
from plotly.subplots import make_subplots
from shared import (
    AMBER,
    BLUE,
    current_client,
    document,
    glossary,
    header,
    plot,
    table,
    with_label,
)

client = current_client()
summary = document(client, "summary.json")
# Upsample so weeks without data become nulls, which plotly draws as breaks in the line.
weekly = (
    table(client, "weekly_category.csv")
    .with_columns(pl.col("week_start").str.to_date())
    .sort("week_start")
    .upsample("week_start", every="1w")
)
optimizer = document(client, "optimizer.json")["margin"]
backtest = table(client, "backtest_summary.csv").filter(pl.col("segment") == "all")
elasticities = with_label(
    table(client, "elasticities.csv").filter(pl.col("model") == "M3 hierarchical")
)

header(
    summary["display_name"],
    f"Weekly scanner data from {summary['stores']} stores, weeks starting "
    f"{summary['first_week']} to {summary['last_week']}. Start here, then use the simulator "
    "to test a price change or the recommendations page for the optimized prices.",
)

last_year = weekly.drop_nulls("revenue").tail(52)
cols = st.columns(4)
cols[0].metric(
    "Revenue per week",
    f"${last_year['revenue'].mean():,.0f}",
    help="Whole category, all stores, average of the last 52 weeks. The line shows the last two years.",
    chart_data=weekly.drop_nulls("revenue")["revenue"].tail(104).to_list(),
    chart_type="area",
    border=True,
)
cols[1].metric(
    "Units sold on a price cut",
    f"{last_year['promo_unit_share'].mean():.0%}",
    help="Share of category units sold in price cut weeks, last 52 weeks.",
    chart_data=weekly.drop_nulls("revenue")["promo_unit_share"].tail(104).to_list(),
    border=True,
)
cols[2].metric(
    "Products modelled",
    summary["products"],
    help="The best-selling products that sold in most weeks. Everything else is left out.",
    border=True,
)
cols[3].metric(
    "Stores",
    summary["stores"],
    help="Stores that reported sales in the analysis window.",
    border=True,
)

st.subheader("What the models found")
m3 = backtest.filter(pl.col("model") == "M3 hierarchical").row(0, named=True)
naive = backtest.filter(pl.col("model") == "M0 naive").row(0, named=True)
top = elasticities.sort("own_elasticity").row(0, named=True)
band = optimizer["error_band_weekly"]
gain = optimizer["weekly_gain"]

cards = st.columns(3)
with cards[0].container(border=True, height="stretch"):
    st.markdown(":material/trending_down: **Most price-sensitive product**")
    st.markdown(
        f"### {top['product']}\nA 1% regular price rise loses about "
        f"**{-top['own_elasticity']:.1f}%** of its weekly units."
    )
    st.page_link(
        "views/elasticities.py", label="See every product", icon=":material/arrow_forward:"
    )
with cards[1].container(border=True, height="stretch"):
    st.markdown(":material/price_check: **Optimized prices**")
    st.markdown(
        f"### {'+' if gain >= 0 else '-'}\\${abs(gain):,.0f} margin a week\n"
        f"{optimizer['gain_pct']:+.1%} versus today. "
        + (
            f"The forecast error band is ±\\${band:,.0f}, so test it in a few stores first."
            if band > abs(gain)
            else f"Larger than the ±\\${band:,.0f} forecast error band."
        )
    )
    st.page_link("views/optimizer.py", label="See the price list", icon=":material/arrow_forward:")
with cards[2].container(border=True, height="stretch"):
    st.markdown(":material/query_stats: **How good are the forecasts?**")
    st.markdown(
        f"### {m3['wape_chain_week']:.0%} error on weekly category units\n"
        f"The pricing model, on a year of held-out weeks. A flat forecast of recent "
        f"regular-price sales misses by {naive['wape_chain_week']:.0%}."
    )
    st.page_link("views/simulator.py", label="Try a scenario", icon=":material/arrow_forward:")

if not (summary["has_feature"] and summary["has_display"]):
    st.info(
        "This retailer's data has no feature or display flags, so the effect of ads and "
        "in-store displays can't be measured. Those controls are switched off in the simulator.",
        icon=":material/info:",
    )

st.subheader("Category sales, week by week")
fig = make_subplots(
    rows=2, cols=1, shared_xaxes=True, row_heights=[0.65, 0.35], vertical_spacing=0.06
)
fig.add_trace(
    go.Scatter(
        x=weekly["week_start"],
        y=weekly["revenue"],
        name="Category revenue ($ per week)",
        line={"color": BLUE, "width": 2},
        hovertemplate="%{y:$,.0f}<extra>Revenue</extra>",
    ),
    row=1,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=weekly["week_start"],
        y=weekly["promo_unit_share"],
        name="Share of units sold on a price cut",
        line={"color": AMBER, "width": 1.5},
        fill="tozeroy",
        hovertemplate="%{y:.0%}<extra>On a price cut</extra>",
    ),
    row=2,
    col=1,
)
fig.update_yaxes(title_text="Revenue ($)", tickprefix="$", tickformat="~s", row=1, col=1)
fig.update_yaxes(title_text="On a price cut", tickformat=".0%", rangemode="tozero", row=2, col=1)
fig.update_xaxes(title_text="Week starting", row=2, col=1)
fig.update_layout(hovermode="x unified")
plot(fig, 460)
gaps = weekly["revenue"].null_count() > 0
st.caption(
    "Revenue spikes line up with weeks when more units sold on a price cut."
    + (" Gaps are weeks with no data from the retailer." if gaps else "")
)

st.subheader("Products")
profile = with_label(table(client, "product_profile.csv")).sort("revenue", descending=True)
st.dataframe(
    profile.select(
        "product",
        pl.col("manufacturer").str.to_titlecase(),
        (pl.col("revenue") / 1e6).round(2),
        "price_median",
        "price_cut_share",
        "mean_depth_on_cut",
    ),
    hide_index=True,
    height=35 * (profile.height + 1) + 3,
    column_config={
        "product": st.column_config.TextColumn("Product", width="large"),
        "manufacturer": "Manufacturer",
        "revenue": st.column_config.NumberColumn(
            "Revenue, whole period",
            format="$%.2fM",
            help="Millions of dollars, all stores, all weeks shown above.",
        ),
        "price_median": st.column_config.NumberColumn("Typical shelf price", format="$%.2f"),
        "price_cut_share": st.column_config.ProgressColumn(
            "Weeks on a price cut",
            format="percent",
            min_value=0,
            max_value=1,
            help="Share of store-weeks where the price was at least 5% below the regular price.",
        ),
        "mean_depth_on_cut": st.column_config.NumberColumn(
            "Average cut", format="percent", help="How far below the regular price, on cut weeks."
        ),
    },
)

with st.expander("Data quality", icon=":material/fact_check:"):
    q = st.columns(3)
    q[0].metric("Store x product x week rows", f"{summary['rows']:,}")
    q[1].metric(
        "Weeks with zero sales",
        f"{summary['zero_sales_share']:.1%}",
        help="Product was on the shelf but sold nothing. Kept in the data as zeros.",
    )
    q[2].metric(
        "Unusual weeks flagged",
        f"{summary['outlier_share']:.1%}",
        help="Regular-price weeks far above the product's usual sales. Kept, but flagged.",
    )
    st.caption("The full data-quality report is in outputs/<client>/data_quality.html.")

glossary("Price cut week", "Price elasticity", "Forecast error band")
