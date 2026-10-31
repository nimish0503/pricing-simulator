import re

import polars as pl
import streamlit as st
from shared import SETTINGS, current_client, document, header, label, maker, table, with_label

client = current_client()
r = document(client, "optimizer.json")["margin"]
rules = SETTINGS.optimizer
header(
    "Which prices should change?",
    "The suggested shelf price for each cereal. The tool picks the prices that make the most "
    "profit for the whole cereal aisle, not just for one box.",
)

units_change = r["optimized_weekly_units"] / r["current_weekly_units"] - 1
cols = st.columns(3)
cols[0].metric("Profit a week, today's prices", f"${r['current_weekly_value']:,.0f}", border=True)
cols[1].metric(
    "Profit a week, new prices",
    f"${r['optimized_weekly_value']:,.0f}",
    f"+${r['weekly_gain']:,.0f} ({r['gain_pct']:+.1%})",
    border=True,
)
cols[2].metric(
    "Boxes sold a week",
    f"{r['optimized_weekly_units']:,.0f}",
    f"{units_change:+.1%}",
    delta_color="off",
    border=True,
    help="Fewer boxes, but each one earns more.",
)
if r["error_band_weekly"] > abs(r["weekly_gain"]):
    st.info(
        f"This is an estimate that could be off by about \\${r['error_band_weekly']:,.0f} a "
        "week. Try the new prices in a few stores first and compare them with similar stores "
        "that kept the old prices.",
        icon=":material/science:",
    )

makers = table(client, "product_clusters.csv").select("product_id", "manufacturer")
prices = (
    with_label(pl.DataFrame(r["prices"]))
    .join(makers.with_columns(pl.col("product_id").cast(pl.String)), on="product_id", how="left")
    .with_columns(
        brand=pl.col("manufacturer").map_elements(maker, return_dtype=pl.String),
        cents=((pl.col("recommended") - pl.col("current")) * 100).round(0).cast(pl.Int64),
        change=(pl.col("recommended") / pl.col("current") - 1) * 100,
    )
    .with_columns(
        what=pl.when(pl.col("cents") > 0)
        .then(pl.lit("▲ Raise by ") + pl.col("cents").cast(pl.String) + "¢")
        .when(pl.col("cents") < 0)
        .then(pl.lit("▼ Lower by ") + (-pl.col("cents")).cast(pl.String) + "¢")
        .otherwise(pl.lit("Keep as is"))
    )
    .sort("change", descending=True)
)
st.subheader("New price list")
st.dataframe(
    prices.select("product", "brand", "current", "recommended", "what", "change"),
    hide_index=True,
    height="content",
    column_config={
        "product": st.column_config.TextColumn("Cereal", width="large"),
        "brand": st.column_config.TextColumn("Made by", width="medium"),
        "current": st.column_config.NumberColumn("Price today", format="$%.2f"),
        "recommended": st.column_config.NumberColumn("New price", format="$%.2f"),
        "what": "What to do",
        "change": st.column_config.NumberColumn("Change", format="%+.1f%%"),
    },
)


def plain(rule: str) -> str:
    for p in r["prices"]:
        rule = rule.replace(f"{p['name']} {p['size_oz']:g} oz", label(p["name"], p["size_oz"]))
    rule = re.sub(
        r" at the top of its band \((.+)\)", r" would go above $\1 if the rules allowed", rule
    )
    rule = re.sub(
        r" at the bottom of its band \((.+)\)", r" would go below $\1 if the rules allowed", rule
    )
    if rule.startswith("category volume"):
        return f"Total boxes sold would fall further than the {rules.max_category_volume_drop:.0%} limit"
    return rule.replace("$", "\\$")


with st.expander("The rules the new prices follow", icon=":material/rule:"):
    st.markdown(
        f"- No price moves more than {rules.max_price_change:.0%} from today\n"
        f"- Prices end in 9 cents\n"
        "- A bigger box of a cereal is cheaper per ounce than a smaller box\n"
        "- Store brands stay cheaper than the name brands they sit next to\n"
        f"- Total cereal boxes sold drop by no more than {rules.max_category_volume_drop:.0%}"
    )
    if r["binding"]:
        st.markdown("These rules are what stops the profit going higher:")
        st.markdown("\n".join(f"- {plain(line)}" for line in r["binding"]))
