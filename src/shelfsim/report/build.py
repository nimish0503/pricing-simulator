"""Client findings report: outputs/<client>/findings.html, written for a pricing manager.

The template fills in tables and charts from output files. The summary and recommendation are
yours: write them in docs/findings/<client>.md (paragraphs separated by blank lines, a line
"## Recommendation" starts the recommendation). Until then the report shows a placeholder.
"""

import json
from datetime import datetime

import numpy as np
import polars as pl
from jinja2 import Environment, PackageLoader, select_autoescape

from shelfsim.config import ROOT, ClientConfig, ModelingConfig
from shelfsim.io import output_dir, read_output
from shelfsim.report.charts import PALETTE, figure, to_base64

MODEL_OF_RECORD = "M3 hierarchical"


def narrative(client: str) -> dict[str, list[str]]:
    path = ROOT / "docs" / "findings" / f"{client}.md"
    if not path.exists():
        return {"summary": [], "recommendation": []}
    summary, _, recommendation = path.read_text().partition("## Recommendation")

    def paragraphs(text):
        return [
            p.strip() for p in text.split("\n\n") if p.strip() and not p.strip().startswith("#")
        ]

    return {"summary": paragraphs(summary), "recommendation": paragraphs(recommendation)}


def build(cfg: ClientConfig, mcfg: ModelingConfig) -> None:
    client = cfg.client_id
    out = output_dir(client)
    elasticities = read_output(client, "elasticities.csv").with_columns(
        pl.col("product_id").cast(pl.String)
    )
    record = elasticities.filter(pl.col("model") == MODEL_OF_RECORD).sort("own_elasticity")
    plausible = record.filter(pl.col("plausible"))
    promo = read_output(client, "promo_plan.csv")
    optimizer = json.loads((out / "optimizer.json").read_text())
    backtest = (out / "backtest.md").read_text() if (out / "backtest.md").exists() else None
    summary = json.loads((ROOT / "outputs" / "deploy" / client / "summary.json").read_text())

    mechanics = (
        promo.group_by("scenario")
        .agg(
            pl.col("incremental_units").mean(),
            pl.col("own_cannibalization_units").mean(),
            pl.col("competitor_loss_units").mean(),
            pl.col("category_incremental_units").mean(),
            pl.col("category_incremental_revenue").mean(),
            pl.col("category_incremental_margin").mean(),
            pl.col("note").first(),
        )
        .with_columns(
            cannibalized_share=pl.col("own_cannibalization_units") / pl.col("incremental_units")
        )
        .sort("category_incremental_margin", descending=True)
    )
    measurable = mechanics.filter(pl.col("note").fill_null("") == "")
    objective = optimizer["default_objective"]
    opt = optimizer[objective]
    prices = (
        pl.DataFrame(opt["prices"])
        .with_columns(change=pl.col("recommended") / pl.col("current") - 1)
        .filter(pl.col("change").abs() > 1e-9)
        .sort(pl.col("change").abs(), descending=True)
    )

    env = Environment(
        loader=PackageLoader("shelfsim.report", "templates"), autoescape=select_autoescape()
    )
    html = env.get_template("findings.html.j2").render(
        client_name=cfg.display_name,
        generated=datetime.now().strftime("%Y-%m-%d %H:%M"),
        data=summary,
        narrative=narrative(client),
        model_of_record=MODEL_OF_RECORD,
        train=(record["train_start"][0], record["train_end"][0]),
        sensitive=plausible.head(3).to_dicts(),
        least_sensitive=plausible.tail(3).reverse().to_dicts(),
        flagged=record.filter(~pl.col("plausible")).to_dicts(),
        plausibility=mcfg.plausibility,
        elasticity_chart=_elasticity_chart(record),
        mechanics=mechanics.to_dicts(),
        best=measurable.head(1).to_dicts(),
        worst=measurable.tail(1).to_dicts(),
        promo_chart=_promo_chart(mechanics),
        objective=objective,
        opt=opt,
        other=optimizer["margin" if objective == "revenue" else "revenue"],
        biggest_moves=prices.head(10).to_dicts(),
        moved=prices.height,
        backtest=backtest,
        limitations=_limitations(cfg, summary, mcfg),
    )
    (out / "findings.html").write_text(html)


def _elasticity_chart(record: pl.DataFrame) -> str:
    data = record.with_columns(
        label=pl.col("name") + " " + pl.col("size_oz").cast(pl.String) + " oz"
    )
    fig, ax = figure(7.5, 0.26 * data.height + 1.2)
    y = np.arange(data.height)
    colors = [PALETTE[0] if p else PALETTE[1] for p in data["plausible"]]
    ax.barh(y, data["own_elasticity"], color=colors)
    ax.errorbar(
        data["own_elasticity"],
        y,
        xerr=[
            data["own_elasticity"] - data["own_lower"],
            data["own_upper"] - data["own_elasticity"],
        ],
        fmt="none",
        ecolor="#444",
        lw=0.8,
    )
    ax.set_yticks(y, data["label"], fontsize=7)
    ax.invert_yaxis()
    ax.axvline(0, color="#999", lw=0.8)
    ax.set_xlabel(
        "Own-price elasticity of regular price (% change in units per 1% price rise), 95% interval"
    )
    ax.set_title("Price sensitivity by product (orange: outside the plausible range)", fontsize=10)
    return to_base64(fig)


def _promo_chart(mechanics: pl.DataFrame) -> str:
    fig, ax = figure(7, 3.2)
    y = np.arange(mechanics.height)
    ax.barh(
        y - 0.2,
        mechanics["category_incremental_revenue"],
        height=0.4,
        color=PALETTE[0],
        label="category revenue",
    )
    ax.barh(
        y + 0.2,
        mechanics["category_incremental_margin"],
        height=0.4,
        color=PALETTE[2],
        label="category gross margin",
    )
    ax.set_yticks(y, mechanics["scenario"], fontsize=8)
    ax.invert_yaxis()
    ax.axvline(0, color="#999", lw=0.8)
    ax.set_xlabel("Change per promo week vs no promo, $ across all stores (mean over products)")
    ax.legend(frameon=False, fontsize=8)
    return to_base64(fig)


def _limitations(cfg: ClientConfig, summary: dict, mcfg: ModelingConfig) -> list[str]:
    items = []
    if not summary["has_feature"] or not summary["has_display"]:
        items.append(
            "The client's data has no feature or display flags, so those mechanics can't be measured and their effect is set to zero."
        )
    if cfg.cost.unit_cost_expr:
        items.append(
            "Unit cost comes from the client's gross margin field, which is average acquisition cost of inventory, not replacement cost. Margin effects of short promos are approximate."
        )
    if cfg.cost.margin_national is not None:
        items.append(
            f"The client reports no cost. Margin assumes {cfg.cost.margin_national:.1%} on national brands and {cfg.cost.margin_private_label:.1%} on private label (ADR 0005)."
        )
    items.append(
        "Cost has no manufacturer promo funding in it, so every promo here pays for its own discount. With trade funding the margin columns would look better."
    )
    items.append(
        f"Market size is {mcfg.choice.market_peak_multiple:g}x each store's peak weekly category units (ADR 0006). The choice model's category-level price response depends on it, and it is an assumption, not a measurement."
    )
    items.append(
        "Elasticities are estimated by OLS-type regressions that treat price as set independently of demand. If the retailer cut prices when demand was weak, the true sensitivity is higher than shown."
    )
    items.append(
        f"Only the top {cfg.selection.top_n_products} national and top {cfg.selection.private_label_top_n} private-label products by revenue are modelled; the rest of the category is part of the outside option."
    )
    return items
