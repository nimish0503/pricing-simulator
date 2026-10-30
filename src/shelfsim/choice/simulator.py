"""Virtual-shopper simulator.

For each store cluster, n shoppers each get their own price sensitivity
alpha_i ~ Normal(alpha_cluster, sd_ratio * |alpha_cluster|). Shopper i's utility for product j
is delta_j + alpha_i * price_j + alpha_cut * discount_j + beta_f * feature_j + beta_d * display_j
+ Gumbel noise, and 0 + Gumbel noise for buying nothing. They pick the highest.

Units come from each shopper's exact logit probabilities averaged over shoppers, which is smooth
in price and is what the optimizer needs. Switching ("where did the volume go") comes from the
simulated choices with the same Gumbel draws before and after the change, so every switch is
one shopper changing their pick.
"""

from dataclasses import asdict, dataclass

import numpy as np
import polars as pl


@dataclass
class ClusterParams:
    cluster: int
    name: str
    alpha: float
    alpha_cut: float
    beta_feature: float
    beta_display: float
    delta: list[float]  # per product, includes the cluster's mean store effect and season
    market_size: float  # shoppers per week across the cluster's stores
    stores: int


@dataclass
class Scenario:
    """Per product: permanent price change (+0.05 = 5% up), temporary cut depth, promo flags."""

    price_change: dict[str, float]
    price_cut: dict[str, float]
    feature: dict[str, bool]
    display: dict[str, bool]

    @classmethod
    def empty(cls) -> "Scenario":
        return cls({}, {}, {}, {})


class Simulator:
    def __init__(
        self,
        products: pl.DataFrame,
        clusters: list[ClusterParams],
        sd_ratio: float,
        n_shoppers: int,
        seed: int,
    ):
        """products: product_id, name, manufacturer, is_private_label, size_oz, cluster,
        base_price (current regular price), unit_cost."""
        self.products = products
        self.ids = products["product_id"].to_list()
        self.base_price = products["base_price"].to_numpy()
        self.cost = products["unit_cost"].fill_null(0.0).to_numpy()
        self.manufacturer = products["manufacturer"].to_numpy()
        self.clusters = clusters
        self.sd_ratio, self.n_shoppers, self.seed = sd_ratio, n_shoppers, seed
        rng = np.random.default_rng(seed)
        self.alpha_i = [
            c.alpha + sd_ratio * abs(c.alpha) * rng.standard_normal(n_shoppers) for c in clusters
        ]
        self.gumbel = [rng.gumbel(size=(n_shoppers, len(self.ids) + 1)) for _ in clusters]

    def to_dict(self) -> dict:
        return {
            "products": self.products.to_dicts(),
            "clusters": [asdict(c) for c in self.clusters],
            "sd_ratio": self.sd_ratio,
            "n_shoppers": self.n_shoppers,
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Simulator":
        return cls(
            pl.DataFrame(data["products"]),
            [ClusterParams(**c) for c in data["clusters"]],
            data["sd_ratio"],
            data["n_shoppers"],
            data["seed"],
        )

    def _arrays(self, scenario: Scenario, prices: np.ndarray | None = None):
        price = (
            self.base_price * np.array([1 + scenario.price_change.get(p, 0.0) for p in self.ids])
            if prices is None
            else prices
        )
        cut = np.array([scenario.price_cut.get(p, 0.0) for p in self.ids])
        feature = np.array([float(scenario.feature.get(p, False)) for p in self.ids])
        display = np.array([float(scenario.display.get(p, False)) for p in self.ids])
        return price, price * cut, feature, display

    def _utility(self, c: int, price, discount, feature, display) -> np.ndarray:
        params = self.clusters[c]
        fixed = (
            np.asarray(params.delta)
            + params.alpha_cut * discount
            + params.beta_feature * feature
            + params.beta_display * display
        )
        return fixed[None, :] + self.alpha_i[c][:, None] * price[None, :]

    def expected_units(self, prices: np.ndarray, scenario: Scenario | None = None) -> np.ndarray:
        """Weekly units per product across all store clusters, at the given regular prices."""
        price, discount, feature, display = self._arrays(scenario or Scenario.empty(), prices)
        total = np.zeros(len(self.ids))
        for c, params in enumerate(self.clusters):
            u = self._utility(c, price, discount, feature, display)
            top = np.maximum(u.max(axis=1, keepdims=True), 0.0)
            expu = np.exp(u - top)
            probs = expu / (np.exp(-top) + expu.sum(axis=1, keepdims=True))
            total += probs.mean(axis=0) * params.market_size
        return total

    def choices(self, scenario: Scenario) -> list[np.ndarray]:
        """Simulated pick per shopper per cluster: -1 is the outside option."""
        price, discount, feature, display = self._arrays(scenario)
        picks = []
        for c in range(len(self.clusters)):
            u = (
                np.hstack(
                    [
                        np.zeros((self.n_shoppers, 1)),
                        self._utility(c, price, discount, feature, display),
                    ]
                )
                + self.gumbel[c]
            )
            picks.append(u.argmax(axis=1) - 1)
        return picks

    def run(self, scenario: Scenario) -> dict:
        base = Scenario.empty()
        before = self.expected_units(self.base_price, base)
        price, discount, *_ = self._arrays(scenario)
        after = self.expected_units(price, scenario)
        paid_before, paid_after = self.base_price, price - discount
        table = self.products.select("product_id", "name", "manufacturer").with_columns(
            base_price=pl.Series(self.base_price),
            scenario_price=pl.Series(paid_after),
            base_units=pl.Series(before),
            scenario_units=pl.Series(after),
            base_revenue=pl.Series(before * paid_before),
            scenario_revenue=pl.Series(after * paid_after),
            base_margin=pl.Series(before * (paid_before - self.cost)),
            scenario_margin=pl.Series(after * (paid_after - self.cost)),
        )
        focal = {
            p
            for d in (scenario.price_change, scenario.price_cut, scenario.feature, scenario.display)
            for p, v in d.items()
            if v
        }
        return {"products": table, "switching": self.switching(scenario, focal)}

    def switching(self, scenario: Scenario, focal: set[str]) -> dict:
        """Units per week that the changed products lose to / win from each source.

        own: other products of the same manufacturer as the changed product
        competitors: other manufacturers' products
        outside: buying nothing in the category (lost) or new category buyers (won)
        """
        flows = {
            "lost_to": {"own": 0.0, "competitors": 0.0, "outside": 0.0},
            "won_from": {"own": 0.0, "competitors": 0.0, "outside": 0.0},
        }
        if not focal:
            return flows
        is_focal = np.array([p in focal for p in self.ids])
        for c, (b, a) in enumerate(
            zip(self.choices(Scenario.empty()), self.choices(scenario), strict=True)
        ):
            weight = self.clusters[c].market_size / self.n_shoppers
            moved = b != a
            for before, after in zip(b[moved], a[moved], strict=True):
                if before >= 0 and is_focal[before] and (after < 0 or not is_focal[after]):
                    kind = (
                        "outside"
                        if after < 0
                        else "own"
                        if self.manufacturer[after] == self.manufacturer[before]
                        else "competitors"
                    )
                    flows["lost_to"][kind] += weight
                elif after >= 0 and is_focal[after] and (before < 0 or not is_focal[before]):
                    kind = (
                        "outside"
                        if before < 0
                        else "own"
                        if self.manufacturer[before] == self.manufacturer[after]
                        else "competitors"
                    )
                    flows["won_from"][kind] += weight
        return flows
