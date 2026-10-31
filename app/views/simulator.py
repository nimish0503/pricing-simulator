import plotly.graph_objects as go
import polars as pl
import streamlit as st
from shared import (
    BLUE,
    GREEN,
    GREY,
    RED,
    SLATE,
    current_client,
    document,
    glossary,
    header,
    maker,
    maker_colors,
    money,
    plot,
    simulator,
    with_label,
)

from shelfsim.choice.simulator import Scenario

client = current_client()
sim = simulator(client)
summary = document(client, "summary.json")
header(
    "Scenario builder",
    "Start from a preset or edit the price list directly, then see the weekly effect on the "
    "whole category, where the volume moves and how the shelf's price ladder changes. Save "
    "scenarios to compare them side by side.",
)

products = with_label(sim.products).with_columns(
    maker=pl.col("manufacturer").map_elements(maker, return_dtype=pl.String),
    base_units=pl.Series(sim.run(Scenario.empty())["products"]["base_units"]),
)
recommended = {
    p["product_id"]: p["recommended"] / p["current"] - 1
    for p in document(client, "optimizer.json")["margin"]["prices"]
}
top3 = set(products.sort("base_units", descending=True)["product_id"].head(3))
PRESETS = {
    "Today's prices": lambda r: (0, 0),
    "Recommended prices": lambda r: (round(recommended.get(r["product_id"], 0) * 100), 0),
    "Private label +5%": lambda r: (5 if r["is_private_label"] else 0, 0),
    "National brands -5%": lambda r: (0 if r["is_private_label"] else -5, 0),
    "Promo the top 3 sellers": lambda r: (0, 20 if r["product_id"] in top3 else 0),
}
preset = st.segmented_control(
    "Start from",
    list(PRESETS),
    default="Recommended prices",
    required=True,
    key=f"preset_{client}",
    help="Fills the price list below. You can then edit any row.",
)
start = products.with_columns(
    change=pl.struct(products.columns).map_elements(
        lambda r: PRESETS[preset](r)[0], return_dtype=pl.Int64
    ),
    cut=pl.struct(products.columns).map_elements(
        lambda r: PRESETS[preset](r)[1], return_dtype=pl.Int64
    ),
    feature=pl.lit(False),
    display=pl.lit(False),
).select("product_id", "product", "maker", "base_price", "change", "cut", "feature", "display")

edit_col, result_col = st.columns([1.5, 1], gap="medium")
with edit_col:
    st.markdown("##### :material/edit_note: Price list")
    edited = st.data_editor(
        start,
        key=f"editor_{client}_{preset}",
        hide_index=True,
        height=35 * (start.height + 1) + 3,
        column_order=["product", "base_price", "change", "cut", "feature", "display"],
        disabled=["product", "maker", "base_price"],
        column_config={
            "product": st.column_config.TextColumn("Product", width="medium"),
            "base_price": st.column_config.NumberColumn("Today", format="$%.2f", width="small"),
            "change": st.column_config.NumberColumn(
                "Change",
                width="small",
                min_value=-20,
                max_value=20,
                step=1,
                format="%+d%%",
                help="Permanent change to the regular price.",
            ),
            "cut": st.column_config.SelectboxColumn(
                "Promo %",
                width="small",
                options=[0, 5, 10, 15, 20, 25, 30],
                help="A one-week temporary cut on top of the regular price.",
            ),
            "feature": st.column_config.CheckboxColumn(
                "Ad",
                width="small",
                disabled=not summary["has_feature"],
                help="In the weekly ad"
                + ("" if summary["has_feature"] else ". Not in this retailer's data."),
            ),
            "display": st.column_config.CheckboxColumn(
                "Display",
                width="small",
                disabled=not summary["has_display"],
                help="On an in-store display"
                + ("" if summary["has_display"] else ". Not in this retailer's data."),
            ),
        },
    )

edited = pl.DataFrame(edited) if not isinstance(edited, pl.DataFrame) else edited
scenario = Scenario(
    {r["product_id"]: (r["change"] or 0) / 100 for r in edited.iter_rows(named=True)},
    {r["product_id"]: (r["cut"] or 0) / 100 for r in edited.iter_rows(named=True)},
    {r["product_id"]: bool(r["feature"]) for r in edited.iter_rows(named=True)},
    {r["product_id"]: bool(r["display"]) for r in edited.iter_rows(named=True)},
)
result = sim.run(scenario)
t = result["products"]
totals = {
    k: (float(t[f"base_{k}"].sum()), float(t[f"scenario_{k}"].sum()))
    for k in ("units", "revenue", "margin")
}
changed = [
    r["product_id"]
    for r in edited.iter_rows(named=True)
    if r["change"] or r["cut"] or r["feature"] or r["display"]
]

with result_col:
    st.markdown("##### :material/monitoring: Weekly effect, whole category")
    m = st.columns(3)
    for col, key, title, fmt in [
        (m[0], "units", "Units", "{:,.1f}k"),
        (m[1], "revenue", "Revenue", "${:,.1f}k"),
        (m[2], "margin", "Margin", "${:,.1f}k"),
    ]:
        base, after = totals[key]
        delta = after - base
        col.metric(
            title,
            fmt.format(after / 1000),
            f"{delta / base:+.1%}" if abs(delta) >= 0.5 else None,
            border=True,
            help=f"Per week, all stores. Today: {fmt.format(base / 1000)}.",
        )
    if not changed:
        st.info(
            "Pick a preset or edit a row in the price list to see the effect.",
            icon=":material/touch_app:",
        )
    else:
        d_units = totals["units"][1] - totals["units"][0]
        d_margin = totals["margin"][1] - totals["margin"][0]
        d_revenue = totals["revenue"][1] - totals["revenue"][0]
        st.markdown(
            f"Changing **{len(changed)} product{'s' if len(changed) > 1 else ''}** "
            f"{'adds' if d_units >= 0 else 'loses'} **{abs(d_units):,.0f} category units** a "
            f"week, moves revenue by **{money(d_revenue, True)}** and margin by "
            f"**{money(d_margin, True)}**."
        )
        flows = result["switching"]
        focal = t.filter(pl.col("product_id").is_in(changed))
        net = float((focal["scenario_units"] - focal["base_units"]).sum())
        if net < 0:
            steps = [
                ("Same maker", -flows["lost_to"]["own"]),
                ("Competitors", -flows["lost_to"]["competitors"]),
                ("Left category", -flows["lost_to"]["outside"]),
            ]
        else:
            steps = [
                ("Same maker", flows["won_from"]["own"]),
                ("Competitors", flows["won_from"]["competitors"]),
                ("New buyers", flows["won_from"]["outside"]),
            ]
        fig = go.Figure(
            go.Waterfall(
                x=[s[0] for s in steps] + ["Net"],
                y=[s[1] for s in steps] + [None],
                measure=["relative"] * len(steps) + ["total"],
                texttemplate="%{delta:+,.0f}",
                textposition="outside",
                increasing={"marker": {"color": GREEN}},
                decreasing={"marker": {"color": RED}},
                totals={"marker": {"color": BLUE}},
                connector={"line": {"color": "rgba(148,163,184,.6)", "dash": "dot"}},
                hovertemplate="%{x}: %{delta:+,.0f} units a week<extra></extra>",
            )
        )
        fig.update_yaxes(title="Units per week, all stores", tickformat=",")
        st.markdown("Where the changed products' volume came from or went:")
        plot(fig, 320, legend=False)

    name = st.text_input(
        "Scenario name", value=preset, key=f"name_{client}", placeholder="Name this scenario"
    )
    if st.button("Save to compare", icon=":material/bookmark_add:", disabled=not changed):
        saved = st.session_state.setdefault(f"saved_{client}", {})
        saved[name] = {
            "Units": totals["units"][1] - totals["units"][0],
            "Revenue": totals["revenue"][1] - totals["revenue"][0],
            "Margin": totals["margin"][1] - totals["margin"][0],
            "Products changed": len(changed),
        }
        st.toast(f"Saved '{name}'", icon=":material/check:")

st.subheader("The shelf's price ladder")
st.caption(
    "Price per ounce for every product, grouped by cluster. Hollow circles are today, filled "
    "dots are the scenario. Shoppers compare within a cluster, so watch the order there."
)
ladder = (
    t.join(
        products.select("product_id", "product", "maker", "size_oz", "cluster_name"),
        on="product_id",
    )
    .with_columns(
        today=pl.col("base_price") / pl.col("size_oz"),
        after=pl.col("scenario_price") / pl.col("size_oz"),
        cluster_name=pl.col("cluster_name").str.slice(0, 1).str.to_uppercase()
        + pl.col("cluster_name").str.slice(1),
    )
    .sort("cluster_name", "today")
)
colors = maker_colors(client)
fig = go.Figure()
for row in ladder.iter_rows(named=True):
    if abs(row["after"] - row["today"]) > 1e-6:
        fig.add_trace(
            go.Scatter(
                x=[row["today"], row["after"]],
                y=[row["cluster_name"]] * 2,
                mode="lines",
                line={"color": GREY, "width": 1.5},
                showlegend=False,
                hoverinfo="skip",
            )
        )
for name_, group in ladder.group_by("maker", maintain_order=True):
    colour = colors.get(name_[0], SLATE)
    fig.add_trace(
        go.Scatter(
            x=group["today"],
            y=group["cluster_name"],
            mode="markers",
            marker={"color": "rgba(0,0,0,0)", "size": 12, "line": {"color": colour, "width": 2}},
            name=name_[0],
            legendgroup=name_[0],
            text=group["product"],
            hovertemplate="%{text}<br>Today: %{x:$.3f}/oz<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=group["after"],
            y=group["cluster_name"],
            mode="markers",
            marker={"color": colour, "size": 9},
            name=name_[0],
            legendgroup=name_[0],
            showlegend=False,
            text=group["product"],
            hovertemplate="%{text}<br>Scenario: %{x:$.3f}/oz<extra></extra>",
        )
    )
fig.update_xaxes(title="Price per ounce", tickprefix="$", tickformat=".2f")
fig.update_yaxes(title=None)
plot(fig, 90 * ladder["cluster_name"].n_unique() + 140)

saved = st.session_state.get(f"saved_{client}", {})
if saved:
    st.subheader("Saved scenarios")
    compare = pl.DataFrame([{"Scenario": k, **v} for k, v in saved.items()])
    fig = go.Figure()
    for metric, colour in [("Revenue", BLUE), ("Margin", "#D97706")]:
        fig.add_trace(
            go.Bar(
                x=compare["Scenario"],
                y=compare[metric],
                name=f"Change in weekly {metric.lower()}",
                marker_color=colour,
                texttemplate="%{y:$,.0f}",
                textposition="outside",
                hovertemplate="%{x}: %{y:$,.0f}<extra>" + metric + "</extra>",
            )
        )
    fig.update_layout(barmode="group")
    fig.update_yaxes(title="Change per week, all stores", tickformat="$~s")
    plot(fig, 340)
    st.dataframe(
        compare.with_columns(pl.col("Units", "Revenue", "Margin").round(0)),
        hide_index=True,
        column_config={
            "Units": st.column_config.NumberColumn("Change in units", format="localized"),
            "Revenue": st.column_config.NumberColumn("Change in revenue", format="dollar"),
            "Margin": st.column_config.NumberColumn("Change in margin", format="dollar"),
        },
    )
    if st.button("Clear saved scenarios", icon=":material/delete_sweep:"):
        st.session_state[f"saved_{client}"] = {}
        st.rerun()

with st.expander("Effect on every product", icon=":material/table_rows:"):
    view = (
        t.join(products.select("product_id", "product"), on="product_id")
        .with_columns(
            units_change=pl.col("scenario_units") - pl.col("base_units"),
            margin_change=pl.col("scenario_margin") - pl.col("base_margin"),
        )
        .sort(pl.col("units_change").abs(), descending=True)
    )
    st.dataframe(
        view.select(
            "product",
            "scenario_price",
            pl.col("base_units", "units_change", "margin_change").round(0),
        ),
        hide_index=True,
        column_config={
            "product": st.column_config.TextColumn("Product", width="large"),
            "scenario_price": st.column_config.NumberColumn("Price paid", format="$%.2f"),
            "base_units": st.column_config.NumberColumn("Units per week today", format="localized"),
            "units_change": st.column_config.NumberColumn("Change in units", format="localized"),
            "margin_change": st.column_config.NumberColumn("Change in margin", format="dollar"),
        },
    )

glossary("Gross margin", "Feature", "Display", "Product cluster")
