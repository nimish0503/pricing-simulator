"""Pydantic models for configs/clients/<client>.yaml and configs/modeling.yaml.

Unknown keys are errors, so a typo in a YAML file fails at load time instead of being ignored.
"""

from datetime import date
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "configs"


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Source(Strict):
    path: str
    reader: Literal["csv", "excel", "stata"]
    sheet: str | None = None
    # Excel exports often carry a title block above the header; the header is the first row
    # containing this cell value.
    header_contains: str | None = None
    encoding: str = "utf8"
    # intermediate name -> raw column name
    columns: dict[str, str]


class Calendar(Strict):
    kind: Literal["week_number", "week_end_date", "week_start_date"]
    week1_start: date | None = None


class Sizes(Strict):
    # oz per unit; the size string is "<number><unit>", unit matched by prefix
    unit_to_oz: dict[str, float] = {"oz": 1.0, "o": 1.0, "lb": 16.0, "g": 0.035274}
    default_unit: str = "oz"
    overrides: dict[str, float] = {}


class ProductRules(Strict):
    category_filter: str | None = None
    # product_id prefix -> manufacturer, longest prefix wins
    manufacturer_prefixes: dict[str, str] = {}
    private_label: str
    category_name: str = "cereal"


class StoreRules(Strict):
    segment_map: dict[str, str] = {}
    extra_columns: list[str] = []


class Promo(Strict):
    # SQL expressions over the renamed raw columns. Null means the client doesn't report it.
    feature: str | None = None
    display: str | None = None
    price_cut: str | None = None


class Selection(Strict):
    last_weeks: int
    top_n_products: int
    private_label_top_n: int = 0
    min_weeks_observed: int


class Cost(Strict):
    # Either an expression over raw columns, or margin rates on base price by brand tier.
    unit_cost_expr: str | None = None
    margin_national: float | None = None
    margin_private_label: float | None = None


class ClientConfig(Strict):
    client_id: str
    display_name: str
    raw_dir: str
    sales: Source
    products: Source
    stores: Source
    row_filters: list[str] = []
    calendar: Calendar
    sizes: Sizes = Sizes()
    product_rules: ProductRules
    store_rules: StoreRules = StoreRules()
    promo: Promo
    selection: Selection
    cost: Cost
    product_cluster_names: dict[int, str] = {}
    store_cluster_names: dict[int, str] = {}


class Onboarding(Strict):
    base_price_window_weeks: int
    max_zero_run_weeks: int
    price_cut_min_depth: float
    outlier_mad_k: float


class Split(Strict):
    holdout_weeks: int
    train_window_weeks: int


class Backtest(Strict):
    horizon_weeks: int
    step_weeks: int
    n_folds: int
    price_change_min: float


class Segmentation(Strict):
    product_k_range: tuple[int, int]
    store_k_range: tuple[int, int]
    min_cluster_size: int
    seed: int


class Demand(Strict):
    fourier_terms: int
    exclude_outliers_from_fit: bool
    naive_window_weeks: int
    hb_max_rows_per_product: int
    hb_draws: int
    hb_tune: int
    hb_chains: int
    gbm_lag_weeks: list[int]
    gbm_rounds: int
    gbm_learning_rate: float
    gbm_num_leaves: int
    elasticity_perturbation: float
    seed: int


class Plausibility(Strict):
    own_elasticity_min: float
    own_elasticity_max: float


class Choice(Strict):
    market_peak_multiple: float
    shoppers_per_cluster: int
    alpha_sd_ratio: float
    seed: int


class Optimizer(Strict):
    max_price_change: float
    price_ending_cents: int
    max_category_volume_drop: float
    max_passes: int
    objective: Literal["revenue", "margin"]


class PromoScenario(Strict):
    name: str
    price_cut: float = 0.0
    feature: bool = False
    display: bool = False


class ModelingConfig(Strict):
    onboarding: Onboarding
    split: Split
    backtest: Backtest
    segmentation: Segmentation
    demand: Demand
    plausibility: Plausibility
    choice: Choice
    optimizer: Optimizer
    promo_scenarios: list[PromoScenario] = Field(min_length=1)


def load_client(client: str, config_dir: Path = CONFIG_DIR) -> ClientConfig:
    path = config_dir / "clients" / f"{client}.yaml"
    return ClientConfig.model_validate(yaml.safe_load(path.read_text()))


def load_modeling(config_dir: Path = CONFIG_DIR) -> ModelingConfig:
    return ModelingConfig.model_validate(yaml.safe_load((config_dir / "modeling.yaml").read_text()))
