import plotly.graph_objects as go
import polars as pl
import streamlit as st
from plotly.subplots import make_subplots
from shared import (
    AMBER,
    BLUE,
    GREEN,
    GREY,
    RED,
    current_client,
    document,
    findings,
    glossary,
    hero,
    plot,
    table,
    with_label,
)

client = current_client()
f = findings(client)
summary = document(client, "summary.json")

verdict = (
    f"The gain is smaller than the ±${f['band']:,.0f} weekly forecast error, so trial the "
    "prices in a few stores before rolling them out."
    if f["band"] > abs(f["gain"])
    else f"That's larger than the ±${f['band']:,.0f} weekly forecast error."
)
hero(
    f"Briefing · {f['name']}",
    f"Raise {f['raised']} of {f['products']} regular prices: +${f['gain']:,.0f} margin a week",
    f"The recommended prices add {f['gain_pct']:.1%} to weekly category margin and "
    f"{'cost' if f['units_change_pct'] < 0 else 'add'} {abs(f['units_change_pct']):.1%} of "
    f"category units. {verdict}",
    [
        ("Category revenue per week", f"${f['weekly_revenue']:,.0f}"),
        ("Units sold on a price cut", f"{f['promo_unit_share']:.0%}"),
        ("Forecast error, weekly category units", f"{f['wape_m3']:.0%}"),
        ("Stores · products", f"{f['stores']} · {f['products']}"),
    ],
)
st.page_link(
    "views/optimizer.py", label="See the recommended price list", icon=":material/arrow_forward:"
)

st.subheader("Three things to know")
cards = st.columns(3)


def mini(fig: go.Figure, height: int = 190) -> None:
    fig.update_layout(
        height=height,
        margin={"l": 4, "r": 4, "t": 4, "b": 4},
        showlegend=False,
        font_family="IBM Plex Sans, sans-serif",
    )
    st.plotly_chart(fig, config={"displayModeBar": False})


mechanics = f["mechanics"].with_columns(
    pl.col("scenario").str.slice(0, 1).str.to_uppercase() + pl.col("scenario").str.slice(1)
)
paying = mechanics.filter(pl.col("margin") > 0)["scenario"].to_list()
with cards[0].container(border=True, height="stretch"):
    title = (
        "Every promo mechanic loses margin"
        if not paying
        else f"{paying[0]} pays best; plain price cuts don't"
        if mechanics.filter(pl.col("scenario").str.contains("price cut"))["margin"].max() < 0
        else f"{paying[0]} pays best"
    )
    st.html(f'<div class="card-title"><span class="step">1</span>{title}</div>')
    st.caption(
        "Average extra category margin from a one-week promo, across products. It counts what "
        "the promoted product takes from the rest of the shelf."
    )
    fig = go.Figure(
        go.Bar(
            x=mechanics["margin"],
            y=mechanics["scenario"],
            orientation="h",
            marker_color=[GREEN if m > 0 else RED for m in mechanics["margin"]],
            texttemplate="%{x:$,.0f}",
            textposition="auto",
            hovertemplate="%{y}: %{x:$,.0f} a promo week<extra></extra>",
        )
    )
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(showticklabels=False, showgrid=False)
    mini(fig)
    if not (f["has_feature"] and f["has_display"]):
        st.caption("Ads and displays can't be measured for this retailer.")

top = (
    with_label(table(client, "elasticities.csv"))
    .filter((pl.col("model") == "M3 hierarchical") & pl.col("plausible"))
    .sort("own_elasticity")
    .head(6)
)
with cards[1].container(border=True, height="stretch"):
    most = f["most_sensitive"]
    st.html(
        f'<div class="card-title"><span class="step">2</span>{most["product"]} is the most '
        "price-sensitive</div>"
    )
    st.caption(
        f"A 1% regular price rise loses about {-most['own_elasticity']:.1f}% of its units. "
        f"The typical product loses {-f['median_elasticity']:.1f}%."
    )
    fig = go.Figure(
        go.Bar(
            x=-top["own_elasticity"],
            y=top["product"],
            orientation="h",
            marker_color=BLUE,
            texttemplate="%{x:.1f}%",
            textposition="auto",
            hovertemplate="%{y}: -%{x:.2f}% units per 1% price rise<extra></extra>",
        )
    )
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(showgrid=False, ticksuffix="%")
    mini(fig)
    st.page_link("views/elasticities.py", label="Every product", icon=":material/arrow_forward:")

with cards[2].container(border=True, height="stretch"):
    ratio = f["promo_ratio"]
    st.html(
        f'<div class="card-title"><span class="step">3</span>Shoppers react {ratio:.1f}x more '
        "to a promo than to a permanent price change</div>"
    )
    st.caption(
        "They stock up on deals. Judging a permanent price cut by promo response would "
        f"overstate its volume about {ratio:.1f}x."
    )
    e = table(client, "elasticities.csv").filter(pl.col("model") == "M3 hierarchical")
    medians = [-e["own_elasticity"].median(), -e["promo_elasticity"].median()]
    fig = go.Figure(
        go.Bar(
            x=["Permanent change", "One-week promo"],
            y=medians,
            marker_color=[BLUE, AMBER],
            texttemplate="%{y:.1f}%",
            textposition="auto",
            hovertemplate="%{x}: %{y:.2f}% more units per 1% lower price<extra></extra>",
        )
    )
    fig.update_yaxes(showgrid=False, ticksuffix="%", title="Units per 1% price cut")
    mini(fig)

st.subheader("Category sales, week by week")
# Upsample so weeks without data become nulls, which plotly draws as breaks in the line.
weekly = (
    table(client, "weekly_category.csv")
    .with_columns(pl.col("week_start").str.to_date())
    .sort("week_start")
    .upsample("week_start", every="1w")
)
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
best = weekly.drop_nulls("revenue").sort("revenue").row(-1, named=True)
fig.add_annotation(
    x=best["week_start"],
    y=best["revenue"],
    text=f"Best week: ${best['revenue']:,.0f}<br>{best['promo_unit_share']:.0%} of units on a cut",
    showarrow=True,
    arrowhead=2,
    ax=60,
    ay=-10,
    bgcolor="rgba(255,255,255,.85)",
    bordercolor=GREY,
    font={"color": "#0F172A", "size": 12},
    row=1,
    col=1,
)
missing = weekly.filter(pl.col("revenue").is_null())
if missing.height:
    gaps = missing.with_columns(run=(pl.col("week_start").diff().dt.total_days() != 7).cum_sum())
    for run in (
        gaps.group_by("run")
        .agg(pl.col("week_start").min(), end=pl.col("week_start").max())
        .iter_rows(named=True)
    ):
        fig.add_vrect(
            x0=run["week_start"],
            x1=run["end"],
            fillcolor=GREY,
            opacity=0.15,
            line_width=0,
            annotation_text="No data",
            annotation_position="top left",
        )
fig.update_yaxes(title_text="Revenue ($)", tickprefix="$", tickformat="~s", row=1, col=1)
fig.update_yaxes(title_text="On a price cut", tickformat=".0%", rangemode="tozero", row=2, col=1)
fig.update_xaxes(title_text="Week starting", row=2, col=1)
fig.update_layout(hovermode="x unified")
plot(fig, 460)

st.subheader("Products")
st.caption("Click a row to open that product's deep-dive.")
prices = pl.DataFrame(document(client, "optimizer.json")["margin"]["prices"]).select(
    "product_id", change=pl.col("recommended") / pl.col("current") - 1
)
m3 = table(client, "elasticities.csv").filter(pl.col("model") == "M3 hierarchical")
profile = (
    with_label(table(client, "product_profile.csv"))
    .join(m3.select("product_id", "own_elasticity"), on="product_id", how="left")
    .join(prices, on="product_id", how="left")
    .sort("revenue", descending=True)
)
picked = st.dataframe(
    profile.select(
        "product",
        pl.col("manufacturer").str.to_titlecase(),
        (pl.col("revenue") / 1e6).round(2),
        "price_median",
        "price_cut_share",
        "own_elasticity",
        "change",
    ),
    hide_index=True,
    height=35 * (profile.height + 1) + 3,
    on_select="rerun",
    selection_mode="single-row",
    key=f"products_{client}",
    column_config={
        "product": st.column_config.TextColumn("Product", width="medium"),
        "manufacturer": "Maker",
        "revenue": st.column_config.NumberColumn(
            "Revenue", format="$%.2fM", help="Millions of dollars, all stores."
        ),
        "price_median": st.column_config.NumberColumn("Shelf price", format="$%.2f"),
        "price_cut_share": st.column_config.ProgressColumn(
            "On a price cut", format="percent", min_value=0, max_value=1
        ),
        "own_elasticity": st.column_config.NumberColumn(
            "Price sensitivity",
            format="%.2f",
            help="% change in weekly units for a 1% regular price rise (hierarchical Bayes).",
        ),
        "change": st.column_config.NumberColumn(
            "New price", format="percent", help="Recommended change to the regular price."
        ),
    },
)
if picked.selection.rows:
    st.session_state["product"] = profile["product_id"][picked.selection.rows[0]]
    st.switch_page("views/product.py")

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

glossary("Price cut week", "Price elasticity", "Forecast error band")
