import polars as pl
import streamlit as st
from shared import document, pick_client

client = pick_client()
result = document(client, "optimizer.json")
st.title("Price optimizer")
objective = st.radio(
    "Maximize weekly category",
    ["margin", "revenue"],
    horizontal=True,
    index=0 if result["default_objective"] == "margin" else 1,
)
r = result[objective]
st.caption(
    "Rules: every price within ±10% of today's base price and ending in .x9, bigger packs cheaper per ounce than smaller packs of the same cereal, private label below national brands in its cluster, category units down at most 3%."
)

cols = st.columns(3)
cols[0].metric(f"Weekly {objective}, today", f"${r['current_weekly_value']:,.0f}")
cols[1].metric(
    f"Weekly {objective}, recommended",
    f"${r['optimized_weekly_value']:,.0f}",
    f"{r['weekly_gain']:+,.0f} ({r['gain_pct']:+.1%})",
)
cols[2].metric(
    "Category units / week",
    f"{r['optimized_weekly_units']:,.0f}",
    f"{r['optimized_weekly_units'] - r['current_weekly_units']:+,.0f}",
)
if r["wape"]:
    band = r["error_band_weekly"]
    verdict = (
        "wider than the gain: test in a few stores first"
        if band > abs(r["weekly_gain"])
        else "narrower than the gain"
    )
    st.warning(
        f"Forecast error band: ±${band:,.0f} per week ({r['wape']:.0%} WAPE on held-out chain-week category units), {verdict}."
    )
if objective == "revenue":
    st.info(
        "Revenue alone rewards cutting prices: the model treats lost category buyers as easy to win back, so it pushes most prices to the bottom of the band. Margin is the safer objective."
    )

prices = (
    pl.DataFrame(r["prices"])
    .with_columns(change=pl.col("recommended") / pl.col("current") - 1)
    .sort("change")
)
st.dataframe(
    prices.select(
        pl.col("name").alias("Product"),
        pl.col("size_oz").alias("Size (oz)"),
        pl.col("current").alias("Today ($)"),
        pl.col("recommended").alias("Recommended ($)"),
        (pl.col("change") * 100).round(1).alias("Change (%)"),
    ),
    use_container_width=True,
    hide_index=True,
)
st.subheader("Rules that bind")
for line in r["binding"] or ["None: the optimum is inside every rule."]:
    st.markdown(f"- {line}")
if r["broken_today"]:
    st.subheader("Rules already broken at today's prices")
    for line in r["broken_today"]:
        st.markdown(f"- {line}")
st.caption(f"Search: {r['method']}.")
