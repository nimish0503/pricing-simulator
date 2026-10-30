import plotly.graph_objects as go
import polars as pl
import streamlit as st
from shared import (
    GREEN,
    RED,
    SETTINGS,
    current_client,
    document,
    glossary,
    header,
    label,
    plot,
    table,
    with_label,
)

client = current_client()
result = document(client, "optimizer.json")
rules = SETTINGS.optimizer
header(
    "What should we charge, and which promos pay?",
    "Regular prices that maximize weekly category margin within the business rules, and the "
    "weekly payback of each promo mechanic for every product.",
)

prices_tab, promo_tab = st.tabs(
    [":material/sell: Regular prices", ":material/campaign: Promo planner"]
)

with prices_tab:
    objective = st.segmented_control(
        "Maximize",
        ["margin", "revenue"],
        default=result["default_objective"],
        format_func=lambda o: f"Weekly {o}",
        required=True,
        help="Margin is the default. Revenue alone rewards cutting every price, because the "
        "model treats lost category shoppers as easy to win back.",
    )
    r = result[objective]
    units_change = r["optimized_weekly_units"] - r["current_weekly_units"]
    cols = st.columns(4)
    cols[0].metric(
        f"{objective.title()} per week today", f"${r['current_weekly_value']:,.0f}", border=True
    )
    cols[1].metric(
        f"{objective.title()} per week, new prices",
        f"${r['optimized_weekly_value']:,.0f}",
        f"{r['weekly_gain']:+,.0f} ({r['gain_pct']:+.1%})",
        border=True,
    )
    cols[2].metric(
        "Units per week",
        f"{r['optimized_weekly_units']:,.0f}",
        f"{units_change:+,.0f} ({units_change / r['current_weekly_units']:+.1%})",
        border=True,
    )
    if r["wape"]:
        band = r["error_band_weekly"]
        cols[3].metric(
            "Forecast error",
            f"±${band:,.0f}",
            help=f"{r['wape']:.0%} average error on weekly category units in held-out weeks, "
            f"applied to weekly {objective}.",
            border=True,
        )
        if band > abs(r["weekly_gain"]):
            st.warning(
                f"The gain (\\${r['weekly_gain']:,.0f} a week) is smaller than the forecast error "
                f"band (±\\${band:,.0f}). Test these prices in a few stores before rolling out.",
                icon=":material/science:",
            )
        else:
            st.success(
                f"The gain is larger than the ±\\${band:,.0f} forecast error band.",
                icon=":material/check_circle:",
            )
    if objective == "revenue":
        st.info(
            "Revenue alone pushes most prices to the bottom of the allowed band. "
            "Margin is the safer objective.",
            icon=":material/info:",
        )

    prices = (
        with_label(pl.DataFrame(r["prices"]))
        .with_columns(change=pl.col("recommended") / pl.col("current") - 1)
        .sort("change")
    )
    st.subheader("Recommended price changes")
    fig = go.Figure(
        go.Bar(
            x=prices["change"],
            y=prices["product"],
            orientation="h",
            marker_color=[GREEN if c > 0 else RED for c in prices["change"]],
            text=[
                f"${a:.2f} → ${b:.2f}"
                for a, b in zip(prices["current"], prices["recommended"], strict=True)
            ],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="%{y}<br>%{text} (%{x:+.1%})<extra></extra>",
        )
    )
    limit = rules.max_price_change
    # Leave room on each side for the outside labels, but never wider than the rule allows.
    low = min(0.0, prices["change"].min()) - limit * 0.6 * (prices["change"].min() < 0)
    high = max(0.0, prices["change"].max()) + limit * 0.6
    fig.update_xaxes(
        title="Change from today's regular price",
        tickformat="+.0%",
        range=[min(low, -limit * 0.1), high],
        zeroline=True,
    )
    fig.update_yaxes(title=None)
    plot(fig, 26 * prices.height + 120, legend=False)
    st.caption(
        "Green bars are price rises, red bars are cuts. Labels show today's price and the new one."
    )

    left, right = st.columns(2)
    with left.container(border=True, height="stretch"):
        st.markdown("##### :material/rule: Business rules")
        st.markdown(
            f"- Every price within ±{limit:.0%} of today's regular price\n"
            f"- Prices end in .x{rules.price_ending_cents}\n"
            "- Bigger packs cost less per ounce than smaller packs of the same cereal\n"
            "- Private label stays below national brands in its cluster\n"
            f"- Category units fall by at most {rules.max_category_volume_drop:.0%}"
        )
    with right.container(border=True, height="stretch"):
        st.markdown("##### :material/lock: Rules holding prices back")
        st.markdown(
            "\n".join(f"- {line}" for line in r["binding"])
            or "None: the best prices sit inside every rule."
        )
        if r["broken_today"]:
            st.markdown("**Already broken at today's prices** (left as they are):")
            st.markdown("\n".join(f"- {line}" for line in r["broken_today"]))
    st.caption(f"Search method: {r['method']}.")

with promo_tab:
    plan = with_label(table(client, "promo_plan.csv")).with_columns(
        scenario=pl.col("scenario").str.slice(0, 1).str.to_uppercase()
        + pl.col("scenario").str.slice(1)
    )
    st.markdown(
        "Each promo runs for one week on one product. **Category margin** counts what the "
        "product gains minus what it takes from other products, so it's the number to judge "
        "a promo on."
    )
    best = (
        plan.filter(pl.col("note").is_null() | (pl.col("note") == ""))
        .sort("category_incremental_margin", descending=True)
        .group_by("product", maintain_order=True)
        .first()
        .sort("category_incremental_margin", descending=True)
    )
    st.subheader("Best promo for each product")
    st.dataframe(
        best.select(
            "product",
            "scenario",
            pl.col("category_incremental_units", "category_incremental_margin").round(0),
        ),
        hide_index=True,
        height=35 * (best.height + 1) + 3,
        column_config={
            "product": st.column_config.TextColumn("Product", width="large"),
            "scenario": "Best mechanic",
            "category_incremental_units": st.column_config.NumberColumn(
                "Extra category units", format="localized"
            ),
            "category_incremental_margin": st.column_config.NumberColumn(
                "Extra category margin", format="dollar", help="For the promo week, all stores."
            ),
        },
    )

    st.subheader("Compare mechanics for one product")
    products = plan.select("product_id", "name", "size_oz").unique(
        "product_id", maintain_order=True
    )
    chosen = st.selectbox(
        "Product",
        products["product_id"].to_list(),
        format_func=lambda pid: label(*products.filter(pl.col("product_id") == pid).row(0)[1:]),
    )
    one = plan.filter(pl.col("product_id") == chosen).sort("category_incremental_margin")
    fig = go.Figure(
        go.Bar(
            x=one["category_incremental_margin"],
            y=one["scenario"],
            orientation="h",
            marker_color=[GREEN if m > 0 else RED for m in one["category_incremental_margin"]],
            texttemplate="%{x:$,.0f}",
            textposition="outside",
            cliponaxis=False,
            hovertemplate="%{y}: %{x:$,.0f} category margin<extra></extra>",
        )
    )
    fig.update_xaxes(title="Extra category margin in the promo week, all stores", tickprefix="$")
    fig.update_yaxes(title=None)
    plot(fig, 300, legend=False)
    notes = [n for n in one["note"].drop_nulls().unique().to_list() if n]
    for note in notes:
        st.caption(f"Not measurable here: {note.split(': ', 1)[-1]}.")
    st.dataframe(
        one.sort("category_incremental_margin", descending=True).select(
            "scenario",
            pl.col(
                "incremental_units",
                "own_cannibalization_units",
                "competitor_loss_units",
                "category_incremental_units",
                "category_incremental_margin",
            ).round(0),
        ),
        hide_index=True,
        column_config={
            "scenario": "Mechanic",
            "incremental_units": st.column_config.NumberColumn(
                "Extra units, this product", format="localized"
            ),
            "own_cannibalization_units": st.column_config.NumberColumn(
                "Taken from same-maker products", format="localized"
            ),
            "competitor_loss_units": st.column_config.NumberColumn(
                "Taken from competitors", format="localized"
            ),
            "category_incremental_units": st.column_config.NumberColumn(
                "Extra category units", format="localized"
            ),
            "category_incremental_margin": st.column_config.NumberColumn(
                "Extra category margin", format="dollar"
            ),
        },
    )

glossary("Gross margin", "Forecast error band", "Feature", "Display")
