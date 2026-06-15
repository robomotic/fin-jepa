"""
GSCPI — Global Supply Chain Pressure Index (NY Fed, Benigno et al.)

Source: https://www.newyorkfed.org/medialibrary/media/research/policy/gscpi/downloads/gscpi-data.xlsx
Monthly, standardized composite of shipping costs + PMI delivery times.
Publication lag: ~45 days after reference month end.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests
from loguru import logger

GSCPI_URL = (
    "https://www.newyorkfed.org/medialibrary/Research/Interactives/gscpi/"
    "downloads/GSCPI_data.xlsx"
)
GSCPI_CACHE = Path("data/cache/gscpi")
_STALE_DAYS = 30
_HEADERS = {"User-Agent": "Mozilla/5.0 fin-jepa research project"}


def _is_stale(path: Path, max_age_days: int = _STALE_DAYS) -> bool:
    if not path.exists():
        return True
    age = (pd.Timestamp.now() - pd.Timestamp(path.stat().st_mtime, unit="s")).days
    return age >= max_age_days


def fetch_gscpi(
    cache: bool = True,
    cache_dir: Path = GSCPI_CACHE,
) -> pd.Series:
    """Download and parse GSCPI from NY Fed.

    Returns pd.Series with month-start DatetimeIndex named 'GSCPI'.
    Publication lag of 45 days must be applied by pipeline.apply_publication_lags.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    xlsx_path = cache_dir / "gscpi.xlsx"
    parquet_path = cache_dir / "gscpi.parquet"

    if cache and not _is_stale(parquet_path):
        logger.debug("GSCPI cache hit")
        df = pd.read_parquet(parquet_path)
        return df["GSCPI"]

    if _is_stale(xlsx_path):
        logger.info("Downloading GSCPI Excel from NY Fed")
        resp = requests.get(GSCPI_URL, headers=_HEADERS, timeout=60)
        resp.raise_for_status()
        xlsx_path.write_bytes(resp.content)

    s = _parse_gscpi_excel(xlsx_path)

    if cache:
        s.to_frame().to_parquet(parquet_path)

    return s


def _parse_gscpi_excel(path: Path) -> pd.Series:
    """Parse GSCPI Excel. The file has a date column and a GSCPI column."""
    # NY Fed serves this as old XLS format despite the .xlsx URL
    engine = "xlrd"
    try:
        sheets = pd.read_excel(path, sheet_name=None, nrows=0, engine=engine)
    except Exception:
        engine = "openpyxl"
        sheets = pd.read_excel(path, sheet_name=None, nrows=0, engine=engine)

    # Prefer sheet with "data" in name; fall back to first sheet
    sheet_names = list(sheets.keys())
    sheet_name = next(
        (s for s in sheet_names if "data" in s.lower()),
        sheet_names[0],
    )
    raw = pd.read_excel(path, sheet_name=sheet_name, engine=engine)

    raw.columns = [str(c).lower().strip() for c in raw.columns]
    cols = list(raw.columns)

    # Find date column
    date_col = _find_col(cols, ["date", "month", "time", "period"])
    # Find value column
    val_col = _find_col(cols, ["gscpi", "index", "value", "supply"])

    if not date_col or not val_col:
        logger.warning(f"GSCPI columns not found in {path}. Columns: {cols}")
        date_col = cols[0]
        val_col = cols[-1]

    # Drop rows where either date or value is missing (covers metadata/attribution rows)
    index = pd.to_datetime(raw[date_col], errors="coerce")
    values = pd.to_numeric(raw[val_col], errors="coerce")

    s = pd.Series(values.values, index=index, name="GSCPI")
    s = s.dropna()
    s = s.sort_index()
    s = s[~s.index.duplicated(keep="first")]

    # Normalise index to month-start
    s.index = s.index.to_period("M").to_timestamp()
    return s


def _find_col(cols: list[str], candidates: list[str]) -> str | None:
    for col in cols:
        for cand in candidates:
            if cand in col:
                return col
    return None
