from datetime import date, timedelta

import numpy as np
import polars as pl

from shelfsim.choice.logit import fit_logit
from shelfsim.choice.nested import nested_elasticities, nested_shares
from shelfsim.choice.simulator import ClusterParams, Scenario, Simulator

ALPHA = -0.8


def products(n=4) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "product_id": [f"p{i}" for i in range(n)],
            "name": [f"CEREAL {i}" for i in range(n)],
            "manufacturer": ["A", "A", "B", "B"][:n],
            "is_private_label": [False] * n,
            "size_oz": [12.0, 18.0, 15.0, 15.0][:n],
            "cluster": [0, 0, 1, 1][:n],
            "base_price": [3.0, 4.0, 3.5, 2.5][:n],
            "unit_cost": [2.0, 2.8, 2.5, 1.8][:n],
        }
    )


def simulator(sd_ratio: float, n_shoppers: int, delta=(1.0, 0.5, 0.8, 0.2)) -> Simulator:
    cluster = ClusterParams(0, "all", ALPHA, -1.2, 0.3, 0.2, list(delta), 10_000.0, 1)
    return Simulator(products(), [cluster], sd_ratio, n_shoppers, seed=3)


def test_simulated_shares_match_closed_form_logit():
    sim = simulator(sd_ratio=0.0, n_shoppers=200_000)
    picks = sim.choices(Scenario.empty())[0]
    simulated = np.array([(picks == j).mean() for j in range(4)])
    v = np.array([1.0, 0.5, 0.8, 0.2]) + ALPHA * sim.base_price
    closed = np.exp(v) / (1 + np.exp(v).sum())
    assert np.abs(simulated - closed).max() < 0.01
    expected = sim.expected_units(sim.base_price) / 10_000.0
    assert np.allclose(expected, closed)


def test_price_rise_loses_units_and_switching_adds_up():
    sim = simulator(sd_ratio=0.3, n_shoppers=50_000)
    result = sim.run(Scenario({"p0": 0.10}, {}, {}, {}))
    t = result["products"]
    lost = (t["base_units"] - t["scenario_units"])[0]
    assert lost > 0
    others_gain = (t["scenario_units"] - t["base_units"])[1:].sum()
    flows = result["switching"]["lost_to"]
    assert flows["own"] > 0 and flows["competitors"] > 0 and flows["outside"] > 0
    # Simulated switching and expected units agree within simulation noise.
    assert abs(sum(flows.values()) - lost) / lost < 0.15
    assert others_gain < lost


def test_nested_shares_reduce_to_logit_and_elasticity_formulas():
    v = np.array([0.2, -0.1, 0.4])
    nests = np.array([0, 0, 1])
    plain = np.exp(v) / (1 + np.exp(v).sum())
    assert np.allclose(nested_shares(v, nests, 0.0), plain)
    shares = nested_shares(v, nests, 0.6)
    assert shares.sum() < 1
    prices = np.array([3.0, 3.5, 2.5])
    e = nested_elasticities(ALPHA, prices, plain, nests, 0.0)
    assert np.allclose(np.diag(e), ALPHA * prices * (1 - plain))
    assert np.isclose(e[0, 2], -ALPHA * prices[2] * plain[2])
    # With nesting, a same-nest rival responds more than an other-nest one.
    e_nested = nested_elasticities(ALPHA, prices, shares, nests, 0.6)
    assert e_nested[0, 1] > nested_elasticities(ALPHA, prices, shares, nests, 0.0)[0, 1]

    # Numerical check of the own elasticity against the share formula.
    h = 1e-5
    bumped = v.copy()
    bumped[0] += ALPHA * prices[0] * h
    numeric = (np.log(nested_shares(bumped, nests, 0.6)[0]) - np.log(shares[0])) / h
    assert np.isclose(numeric, e_nested[0, 0], rtol=1e-3)


def test_berry_inversion_recovers_alpha():
    rng = np.random.default_rng(0)
    delta = np.array([1.0, 0.5, 0.8, 0.2])
    rows = []
    for s in range(6):
        for t in range(80):
            prices = np.array([3.0, 4.0, 3.5, 2.5]) * np.exp(0.1 * rng.standard_normal(4))
            v = delta + 0.1 * s + ALPHA * prices + 0.05 * rng.standard_normal(4)
            shares = np.exp(v) / (1 + np.exp(v).sum())
            for j in range(4):
                rows.append(
                    {
                        "store_id": str(s),
                        "product_id": f"p{j}",
                        "week_start": date(2010, 1, 4) + timedelta(weeks=t),
                        "units": int(shares[j] * 1_000_000),
                        "base_price": prices[j],
                        "unit_price": prices[j],
                        "feature": 0.0,
                        "display": 0.0,
                    }
                )
    frame = pl.DataFrame(rows)
    sizes = frame.select("store_id").unique().with_columns(market_size=pl.lit(1_000_000.0))
    model = fit_logit(frame, sizes)
    assert abs(model.alpha - ALPHA) < 0.03
    predicted = model.predict_units(frame, sizes)
    assert np.abs(predicted - frame["units"].to_numpy()).sum() / frame["units"].sum() < 0.1


def test_scenario_runs_in_under_two_seconds():
    import time

    n = 28
    rng = np.random.default_rng(0)
    frame = pl.DataFrame(
        {
            "product_id": [f"p{i}" for i in range(n)],
            "name": [f"CEREAL {i}" for i in range(n)],
            "manufacturer": [f"M{i % 5}" for i in range(n)],
            "is_private_label": [False] * n,
            "size_oz": [15.0] * n,
            "cluster": [i % 4 for i in range(n)],
            "base_price": rng.uniform(2, 5, n),
            "unit_cost": rng.uniform(1.5, 3, n),
        }
    )
    clusters = [
        ClusterParams(c, str(c), ALPHA, -1.2, 0.0, 0.0, list(rng.normal(0, 0.5, n)), 50_000.0, 18)
        for c in range(5)
    ]
    sim = Simulator(frame, clusters, 0.3, 10_000, seed=1)
    started = time.perf_counter()
    sim.run(Scenario({"p0": 0.05, "p3": -0.05}, {"p7": 0.2}, {}, {}))
    assert time.perf_counter() - started < 2.0
