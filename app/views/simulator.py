import time

import plotly.graph_objects as go
import polars as pl
import streamlit as st
from shared import (
    BLUE,
    GREEN,
    RED,
    current_client,
    document,
    glossary,
    header,
    plot,
    simulator,
    with_label,
)

from shelfsim.choice.simulator import Scenario

client = current_client()
sim = simulator(client)
summary = document(client, "summary.json")
header(
    "What happens if we change prices?",
    "Pick products, move their prices or add a promo, and see the weekly effect on the whole "
    "category across all stores. Simulated with 10,000 virtual shoppers per store cluster.",
)

products = with_label(sim.products)
by_label = dict(zip(products["product"], products["product_id"], strict=True))


def reset() -> None:
    for key in [k for k in st.session_state if k[:2] in ("p_", "c_", "f_", "d_")]:
        del st.session_state[key]


controls, results = st.columns([1, 1.7], gap="large")
changes, cuts, features, displays = {}, {}, {}, {}
with controls.container(border=True):
    st.markdown("#### :material/tune: Scenario")
    chosen = st.multiselect(
        "Products to change",
        list(by_label),
        default=list(by_label)[:1],
        placeholder="Choose products",
        key=f"chosen_{client}",
    )
    for name in chosen:
        pid = by_label[name]
        st.divider()
        st.markdown(f"**{name}**")
        changes[pid] = (
            st.slider(
                "Permanent price change",
                -10,
                10,
                0,
                format="%d%%",
                key=f"p_{pid}",
                help="A change to the regular shelf price, held for good.",
            )
            / 100
        )
        cuts[pid] = (
            st.slider(
                "One-week promo price cut",
                0,
                30,
                0,
                step=5,
                format="%d%%",
                key=f"c_{pid}",
                help="A temporary cut for one week, on top of the regular price.",
            )
            / 100
        )
        t1, t2 = st.columns(2)
        features[pid] = t1.toggle(
            "In the weekly ad",
            key=f"f_{pid}",
            disabled=not summary["has_feature"],
            help=None if summary["has_feature"] else "This retailer's data has no ad flags.",
        )
        displays[pid] = t2.toggle(
            "In-store display",
            key=f"d_{pid}",
            disabled=not summary["has_display"],
            help=None if summary["has_display"] else "This retailer's data has no display flags.",
        )
    st.button("Reset all changes", icon=":material/restart_alt:", on_click=reset, width="stretch")

started = time.perf_counter()
result = sim.run(Scenario(changes, cuts, features, displays))
elapsed = time.perf_counter() - started
t = result["products"]


def total(column: str, when: str) -> float:
    return float(t[f"{when}_{column}"].sum())


with results:
    cols = st.columns(3)
    for col, column, title, fmt in [
        (cols[0], "units", "Units per week", "{:,.0f}"),
        (cols[1], "revenue", "Revenue per week", "${:,.0f}"),
        (cols[2], "margin", "Margin per week", "${:,.0f}"),
    ]:
        base, scenario = total(column, "base"), total(column, "scenario")
        change = scenario - base
        col.metric(
            title,
            fmt.format(scenario),
            f"{change:+,.0f} ({change / base:+.1%})" if base and abs(change) >= 0.5 else None,
            border=True,
            help=f"Whole category, all stores. Today: {fmt.format(base)}.",
        )

    flows = result["switching"]
    focal = t.filter(pl.col("product_id").is_in([by_label[c] for c in chosen]))
    change = float((focal["scenario_units"] - focal["base_units"]).sum())
    if not chosen or abs(change) < 0.5:
        st.info(
            "Move a slider or switch on a promo to see where the volume goes.",
            icon=":material/touch_app:",
        )
    else:
        margin_change = total("margin", "scenario") - total("margin", "base")
        verb = "gain" if change > 0 else "lose"
        subject = chosen[0] if len(chosen) == 1 else "The changed products"
        if len(chosen) == 1:
            verb += "s"
        st.markdown(
            f"{subject} **{verb} {abs(change):,.0f} units a week**. "
            f"Category margin {'rises' if margin_change >= 0 else 'falls'} by "
            f"**\\${abs(margin_change):,.0f}** a week."
        )
        st.markdown("##### Where the volume came from or went")
        if change < 0:
            steps = [
                ("Lost to same-maker products", -flows["lost_to"]["own"]),
                ("Lost to competitors", -flows["lost_to"]["competitors"]),
                ("Stopped buying in the category", -flows["lost_to"]["outside"]),
            ]
        else:
            steps = [
                ("Won from same-maker products", flows["won_from"]["own"]),
                ("Won from competitors", flows["won_from"]["competitors"]),
                ("New category buyers", flows["won_from"]["outside"]),
            ]
        fig = go.Figure(
            go.Waterfall(
                x=[s[0] for s in steps] + ["Net change"],
                y=[s[1] for s in steps] + [None],
                measure=["relative"] * len(steps) + ["total"],
                texttemplate="%{delta:+,.0f}",
                textposition="outside",
                increasing={"marker": {"color": GREEN}},
                decreasing={"marker": {"color": RED}},
                totals={"marker": {"color": BLUE}},
                connector={"line": {"color": "rgba(148,163,184,.6)", "dash": "dot"}},
                hovertemplate="%{x}: %{delta:+,.0f} units<extra></extra>",
            )
        )
        fig.update_yaxes(title="Units per week, all stores", tickformat=",")
        plot(fig, 360, legend=False)
        st.caption(
            "Switching is counted from simulated shoppers who change their pick, so it carries "
            "a little simulation noise against the totals above. "
            f"Simulated in {elapsed:.2f} s."
        )

st.subheader("Effect on every product")
view = (
    t.join(products.select("product_id", "product"), on="product_id")
    .with_columns(
        units_change=pl.col("scenario_units") - pl.col("base_units"),
        revenue_change=pl.col("scenario_revenue") - pl.col("base_revenue"),
        margin_change=pl.col("scenario_margin") - pl.col("base_margin"),
    )
    .sort(pl.col("units_change").abs(), descending=True)
)
st.dataframe(
    view.select(
        "product",
        "scenario_price",
        pl.col("base_units", "units_change", "revenue_change", "margin_change").round(0),
    ),
    hide_index=True,
    column_config={
        "product": st.column_config.TextColumn("Product", width="large"),
        "scenario_price": st.column_config.NumberColumn(
            "Price paid", format="$%.2f", help="Average shelf price in the scenario week."
        ),
        "base_units": st.column_config.NumberColumn("Units per week today", format="localized"),
        "units_change": st.column_config.NumberColumn("Change in units", format="localized"),
        "revenue_change": st.column_config.NumberColumn("Change in revenue", format="dollar"),
        "margin_change": st.column_config.NumberColumn("Change in margin", format="dollar"),
    },
)

glossary("Gross margin", "Feature", "Display", "Product cluster")
