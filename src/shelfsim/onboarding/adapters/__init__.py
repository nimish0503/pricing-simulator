"""Read one raw source file and rename its columns to the names the common steps expect.

This is the only place that knows about raw file formats. Which file, which reader and which
columns all come from the client's YAML, so a new client in a known format needs no code.
"""

from pathlib import Path

import pandas as pd
import polars as pl

from shelfsim.config import ROOT, Source

ID_COLUMNS = ("store_id", "product_id")


def read_source(source: Source, raw_dir: str, root: Path = ROOT) -> pl.DataFrame:
    path = root / raw_dir / source.path
    if not path.exists():
        raise FileNotFoundError(f"Raw file {path} not found. See the dataset section of the spec.")

    raw_columns = list(source.columns.values())
    match source.reader:
        case "csv":
            frame = pl.read_csv(
                path, encoding=source.encoding, columns=raw_columns, infer_schema_length=100_000
            )
        case "excel":
            frame = pl.from_pandas(_read_excel(path, source))
        case "stata":
            frame = pl.from_pandas(pd.read_stata(path, columns=raw_columns))

    frame = frame.select(pl.col(raw).alias(name) for name, raw in source.columns.items())
    return frame.with_columns(_as_id(c) for c in ID_COLUMNS if c in frame.columns).filter(
        pl.all_horizontal(pl.col(c).is_not_null() for c in ID_COLUMNS if c in frame.columns)
    )


def _read_excel(path: Path, source: Source) -> pd.DataFrame:
    sheet = pd.read_excel(path, sheet_name=source.sheet, header=None)
    header_row = 0
    if source.header_contains:
        matches = sheet.index[sheet.eq(source.header_contains).any(axis=1)]
        if len(matches) == 0:
            raise ValueError(f"No header row containing {source.header_contains!r} in {path}")
        header_row = matches[0]
    frame = sheet.iloc[header_row + 1 :].set_axis(sheet.iloc[header_row], axis=1)
    return frame[list(source.columns.values())].infer_objects()


def _as_id(column: str) -> pl.Expr:
    """IDs become strings, and a float ID like 2.0 (Stata, Excel) becomes "2"."""
    col = pl.col(column)
    as_text = col.cast(pl.String, strict=False)
    return (
        pl.when(as_text.str.contains(r"^\d+\.0+$"))
        .then(as_text.str.replace(r"\.0+$", ""))
        .otherwise(as_text.str.strip_chars())
        .alias(column)
    )
