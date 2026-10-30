"""Promo planner: compare promo mechanics for one product with the simulator."""

import numpy as np
import polars as pl

from shelfsim.choice.simulator import Scenario, Simulator
from shelfsim.config import PromoScenario


def plan(sim: Simulator, product_id: str, scenarios: list[PromoScenario]) -> pl.DataFrame:
    """Weekly effect of each mechanic, run on its own for one week across all stores.

    incremental units / revenue / margin: for the promoted product
    own cannibalization: units the same manufacturer's other products lose
    competitor loss: units other manufacturers' products lose
    category units / margin: net change for the whole category, the retailer's view
    """
    j = sim.ids.index(product_id)
    own = (sim.manufacturer == sim.manufacturer[j]) & (np.arange(len(sim.ids)) != j)
    others = sim.manufacturer != sim.manufacturer[j]
    no_feature = all(c.beta_feature == 0 for c in sim.clusters)
    no_display = all(c.beta_display == 0 for c in sim.clusters)

    rows = []
    for s in scenarios:
        result = sim.run(
            Scenario(
                {}, {product_id: s.price_cut}, {product_id: s.feature}, {product_id: s.display}
            )
        )
        t = result["products"]
        d_units = (t["scenario_units"] - t["base_units"]).to_numpy()
        d_revenue = (t["scenario_revenue"] - t["base_revenue"]).to_numpy()
        d_margin = (t["scenario_margin"] - t["base_margin"]).to_numpy()
        note = []
        if s.feature and no_feature:
            note.append("feature not measurable: client reports no feature data")
        if s.display and no_display:
            note.append("display not measurable: client reports no display data")
        rows.append(
            {
                "product_id": product_id,
                "scenario": s.name,
                "incremental_units": d_units[j],
                "incremental_revenue": d_revenue[j],
                "incremental_margin": d_margin[j],
                "own_cannibalization_units": -d_units[own].sum(),
                "competitor_loss_units": -d_units[others].sum(),
                "category_incremental_units": d_units.sum(),
                "category_incremental_revenue": d_revenue.sum(),
                "category_incremental_margin": d_margin.sum(),
                "note": "; ".join(note),
            }
        )
    return pl.DataFrame(rows)
