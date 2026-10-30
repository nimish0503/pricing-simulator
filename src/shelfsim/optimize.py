"""Price optimizer: the regular prices that maximize category revenue or gross margin, using the
simulator, without breaking any business rule.

Rules (configs/modeling.yaml):
  band      every price within +/- max_price_change of its current base price
  ending    every price ends in .x9
  ladder    a bigger pack of the same product costs less per ounce than a smaller one
  private   private label stays below every national brand in its product cluster, per ounce
  volume    category units drop by at most max_category_volume_drop

Ladder and private-label rules are enforced only for pairs that already satisfy them at today's
prices; pairs that break them today are reported, not silently fixed by the optimizer.

Search: SLSQP on the continuous problem, snapped to the .x9 grid, then coordinate descent over
the grid (try every allowed price for one product with the others fixed, keep the best feasible
move, repeat). Coordinate descent from today's prices also runs, and the better feasible answer
wins, so the grid search is the fallback when SLSQP fails or lands somewhere infeasible.
"""

import re
from dataclasses import dataclass, field

import numpy as np
import polars as pl
from scipy.optimize import minimize

from shelfsim.choice.simulator import Simulator

PPO_TOL = 1e-9


@dataclass
class Rules:
    max_change: float
    ending_cents: int
    max_volume_drop: float
    ladder: list[tuple[int, int]]  # (smaller pack, bigger pack)
    private_label: list[tuple[int, int]]  # (private label, national brand)
    broken_today: list[str] = field(default_factory=list)


@dataclass
class Result:
    status: str
    method: str
    objective: str
    prices: np.ndarray
    current_prices: np.ndarray
    value: float
    current_value: float
    units: float
    current_units: float
    binding: list[str]
    broken_today: list[str]

    @property
    def gain(self) -> float:
        return self.value - self.current_value


def candidates(base: float, max_change: float, ending_cents: int) -> np.ndarray:
    low, high = (
        int(np.ceil(base * (1 - max_change) * 100 - 1e-9)),
        int(np.floor(base * (1 + max_change) * 100 + 1e-9)),
    )
    cents = np.arange(low, high + 1)
    return cents[cents % 10 == ending_cents] / 100.0


def product_line(name: str) -> str:
    """Name without size text, so '12 OZ' and '18 OZ' boxes of the same cereal match."""
    return re.sub(r"\s+", " ", re.sub(r"\d+(\.\d+)?\s*(OZ|O|LB|G)\b", "", name.upper())).strip()


def build_rules(
    products: pl.DataFrame, max_change: float, ending_cents: int, max_volume_drop: float
) -> Rules:
    rows = products.to_dicts()
    ppo = products["base_price"].to_numpy() / products["size_oz"].to_numpy()
    ladder, private, broken = [], [], []
    for i, a in enumerate(rows):
        for j, b in enumerate(rows):
            if i == j:
                continue
            same_line = a["manufacturer"] == b["manufacturer"] and product_line(
                a["name"]
            ) == product_line(b["name"])
            if same_line and a["size_oz"] < b["size_oz"]:
                if ppo[j] < ppo[i]:
                    ladder.append((i, j))
                else:
                    broken.append(
                        f"{a['name']}: {b['size_oz']:g} oz costs more per oz than {a['size_oz']:g} oz"
                    )
            if a["is_private_label"] and not b["is_private_label"] and a["cluster"] == b["cluster"]:
                if ppo[i] < ppo[j]:
                    private.append((i, j))
                else:
                    broken.append(f"private label {a['name']} costs more per oz than {b['name']}")
    return Rules(max_change, ending_cents, max_volume_drop, ladder, private, broken)


def violations(
    prices: np.ndarray,
    base: np.ndarray,
    sizes: np.ndarray,
    rules: Rules,
    units: float,
    min_units: float,
) -> float:
    """Total violation size, 0 when every rule holds."""
    ppo = prices / sizes
    total = np.clip(np.abs(prices / base - 1) - rules.max_change - 1e-9, 0, None).sum()
    total += sum(max(0.0, ppo[big] - ppo[small] + PPO_TOL) for small, big in rules.ladder)
    total += sum(max(0.0, ppo[pl_] - ppo[nb] + PPO_TOL) for pl_, nb in rules.private_label)
    total += max(0.0, min_units - units) / max(min_units, 1.0)
    return total


def optimize(sim: Simulator, rules: Rules, objective: str, max_passes: int) -> Result:
    base = sim.base_price
    sizes = sim.products["size_oz"].to_numpy()
    grid = [candidates(b, rules.max_change, rules.ending_cents) for b in base]

    def value(prices: np.ndarray) -> tuple[float, float]:
        units = sim.expected_units(prices)
        unit_value = prices - sim.cost if objective == "margin" else prices
        return float(units @ unit_value), float(units.sum())

    current_value, current_units = value(base)
    min_units = current_units * (1 - rules.max_volume_drop)

    def bad(prices: np.ndarray) -> float:
        return violations(prices, base, sizes, rules, value(prices)[1], min_units)

    def descend(start: np.ndarray) -> np.ndarray | None:
        prices = start.copy()
        if bad(prices) > 0:
            prices = repair(prices)
            if prices is None:
                return None
        best = value(prices)[0]
        for _ in range(max_passes):
            improved = False
            for j in range(len(prices)):
                for candidate in grid[j]:
                    if candidate == prices[j]:
                        continue
                    trial = prices.copy()
                    trial[j] = candidate
                    trial_value, trial_units = value(trial)
                    if (
                        trial_value > best + 1e-9
                        and violations(trial, base, sizes, rules, trial_units, min_units) == 0
                    ):
                        prices, best, improved = trial, trial_value, True
            if not improved:
                break
        return prices

    def repair(prices: np.ndarray) -> np.ndarray | None:
        for _ in range(max_passes):
            for j in range(len(prices)):
                options = [
                    (bad(np.where(np.arange(len(prices)) == j, c, prices)), c) for c in grid[j]
                ]
                prices = np.where(np.arange(len(prices)) == j, min(options)[1], prices)
            if bad(prices) == 0:
                return prices
        return None

    def snap(prices: np.ndarray) -> np.ndarray:
        return np.array([g[np.abs(g - p).argmin()] for g, p in zip(grid, prices, strict=True)])

    if any(len(g) == 0 for g in grid):
        return _result(
            "infeasible",
            "none",
            objective,
            base,
            base,
            current_value,
            current_value,
            current_units,
            current_units,
            [],
            rules,
        )

    tries = {"grid search": descend(snap(base))}
    slsqp = _slsqp(sim, base, sizes, rules, objective, min_units)
    if slsqp is not None:
        tries["SLSQP + grid polish"] = descend(snap(slsqp))
    feasible = {k: v for k, v in tries.items() if v is not None}
    if not feasible:
        return _result(
            "infeasible",
            "none",
            objective,
            base,
            base,
            current_value,
            current_value,
            current_units,
            current_units,
            [],
            rules,
        )
    method, prices = max(feasible.items(), key=lambda kv: value(kv[1])[0])
    best_value, units = value(prices)
    binding = _binding(prices, sizes, grid, rules, units, min_units, sim)
    return _result(
        "optimal",
        method,
        objective,
        prices,
        base,
        best_value,
        current_value,
        units,
        current_units,
        binding,
        rules,
    )


def _result(
    status, method, objective, prices, base, best, current, units, current_units, binding, rules
) -> Result:
    return Result(
        status,
        method,
        objective,
        prices,
        base,
        best,
        current,
        units,
        current_units,
        binding,
        rules.broken_today,
    )


def _slsqp(
    sim: Simulator, base, sizes, rules: Rules, objective: str, min_units: float
) -> np.ndarray | None:
    def negative(prices):
        units = sim.expected_units(prices)
        return -float(units @ (prices - sim.cost if objective == "margin" else prices)) / 1000.0

    constraints = [{"type": "ineq", "fun": lambda p: sim.expected_units(p).sum() / min_units - 1.0}]
    for small, big in rules.ladder:
        constraints.append(
            {
                "type": "ineq",
                "fun": lambda p, s=small, b=big: p[s] / sizes[s] - p[b] / sizes[b] - 1e-4,
            }
        )
    for pl_, nb in rules.private_label:
        constraints.append(
            {"type": "ineq", "fun": lambda p, a=pl_, b=nb: p[b] / sizes[b] - p[a] / sizes[a] - 1e-4}
        )
    bounds = [(b * (1 - rules.max_change), b * (1 + rules.max_change)) for b in base]
    result = minimize(
        negative,
        base,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 200},
    )
    return result.x if result.success else None


def _binding(prices, sizes, grid, rules: Rules, units, min_units, sim: Simulator) -> list[str]:
    names = sim.products["name"].to_list()
    out = []
    for j, g in enumerate(grid):
        if prices[j] == g.max():
            out.append(f"{names[j]} {sizes[j]:g} oz at the top of its band ({g.max():.2f})")
        elif prices[j] == g.min():
            out.append(f"{names[j]} {sizes[j]:g} oz at the bottom of its band ({g.min():.2f})")
    ppo = prices / sizes
    for small, big in rules.ladder:
        if ppo[small] - ppo[big] < 0.005:
            out.append(
                f"pack ladder {names[small]}: {sizes[big]:g} oz only {100 * (ppo[small] - ppo[big]):.2f} c/oz below {sizes[small]:g} oz"
            )
    for pl_, nb in rules.private_label:
        if ppo[nb] - ppo[pl_] < 0.005:
            out.append(f"private label {names[pl_]} within 0.5 c/oz of {names[nb]}")
    if units < min_units * 1.002:
        out.append(f"category volume at its floor ({rules.max_volume_drop:.0%} below today)")
    return out
