import numpy as np
import polars as pl
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from shelfsim.choice.simulator import ClusterParams, Simulator
from shelfsim.optimize import build_rules, candidates, optimize, product_line


def make_problem(prices, sizes, private, deltas, alpha):
    n = len(prices)
    products = pl.DataFrame(
        {
            "product_id": [f"p{i}" for i in range(n)],
            # Products 0 and 1 are two pack sizes of the same cereal, so the ladder rule applies.
            "name": ["CRUNCH 12 OZ", "CRUNCH 18 OZ", *[f"OTHER {i}" for i in range(2, n)]],
            "manufacturer": ["M", "M", *["PL" if private[i] else f"N{i}" for i in range(2, n)]],
            "is_private_label": [False, False, *private[2:]],
            "size_oz": sizes,
            "cluster": [0] * n,
            "base_price": prices,
            "unit_cost": [p * 0.7 for p in prices],
        }
    )
    sim = Simulator(
        products,
        [ClusterParams(0, "all", alpha, alpha * 1.5, 0.0, 0.0, deltas, 5000.0, 1)],
        0.2,
        500,
        seed=1,
    )
    return products, sim


@st.composite
def problems(draw):
    n = draw(st.integers(3, 5))
    prices = [draw(st.floats(1.5, 6.0)) for _ in range(n)]
    sizes = [12.0, 18.0, *[draw(st.sampled_from([10.0, 14.0, 20.0, 25.0])) for _ in range(n - 2)]]
    private = [False, False, *[draw(st.booleans()) for _ in range(n - 2)]]
    deltas = [draw(st.floats(-1.0, 2.0)) for _ in range(n)]
    alpha = draw(st.floats(-1.5, -0.2))
    drop = draw(st.floats(0.0, 0.1))
    return prices, sizes, private, deltas, alpha, drop


@settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(problems(), st.sampled_from(["revenue", "margin"]))
def test_optimizer_never_breaks_a_constraint(problem, objective):
    prices, sizes, private, deltas, alpha, drop = problem
    products, sim = make_problem(prices, sizes, private, deltas, alpha)
    rules = build_rules(products, 0.10, 9, drop)
    result = optimize(sim, rules, objective, max_passes=3)
    if result.status != "optimal":
        return
    p, base, size = result.prices, np.array(prices), np.array(sizes)
    assert np.all(np.abs(p / base - 1) <= 0.10 + 1e-9)
    assert np.all(np.isclose(np.round(p * 100) % 10, 9))
    ppo = p / size
    for small, big in rules.ladder:
        assert ppo[big] < ppo[small]
    for pl_, nb in rules.private_label:
        assert ppo[pl_] < ppo[nb]
    assert sim.expected_units(p).sum() >= sim.expected_units(base).sum() * (1 - drop) - 1e-6


def test_candidates_end_in_nine_and_stay_in_band():
    grid = candidates(3.35, 0.10, 9)
    # 3.35 +/- 10% is 3.015 to 3.685
    assert list(grid) == [3.09, 3.19, 3.29, 3.39, 3.49, 3.59]


def test_product_line_ignores_size():
    assert product_line("CHEERIOS 15 OZ") == product_line("CHEERIOS 20 OZ") == "CHEERIOS"
