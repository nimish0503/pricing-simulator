"""Where canonical tables and outputs live, and how everything downstream reads them."""

from pathlib import Path

import polars as pl

from shelfsim.config import ROOT

DATA_DIR = ROOT / "data"
DUCKDB_PATH = DATA_DIR / "shelfsim.duckdb"
OUTPUTS_DIR = ROOT / "outputs"


def canonical_dir(client: str) -> Path:
    return DATA_DIR / "canonical" / client


def output_dir(client: str) -> Path:
    path = OUTPUTS_DIR / client
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_table(client: str, table: str) -> pl.DataFrame:
    path = canonical_dir(client) / f"{table}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `make onboard CLIENT={client}` first.")
    return pl.read_parquet(path)


def read_output(client: str, name: str) -> pl.DataFrame:
    path = OUTPUTS_DIR / client / name
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run the earlier pipeline steps first.")
    return pl.read_csv(path) if path.suffix == ".csv" else pl.read_parquet(path)
