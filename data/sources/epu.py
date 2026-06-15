"""
EPU Index downloader — Baker, Bloom, Davis.
License: Creative Commons Attribution 4.0.

US monthly: https://www.policyuncertainty.com/media/US_Policy_Uncertainty_Data.xlsx
Global:     https://www.policyuncertainty.com/media/Global_Policy_Uncertainty_Data.xlsx
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests
from loguru import logger

EPU_US_URL     = "https://www.policyuncertainty.com/media/US_Policy_Uncertainty_Data.xlsx"
EPU_GLOBAL_URL = "https://www.policyuncertainty.com/media/Global_Policy_Uncertainty_Data.xlsx"
EPU_CACHE = Path("data/cache/epu")

_STALE_DAYS = 30
_HEADERS = {"User-Agent": "Mozilla/5.0 fin-jepa research project"}


def _download_excel(url: str, dest: Path) -> None:
    resp = requests.get(url, headers=_HEADERS, timeout=60)
    resp.raise_for_status()
    dest.write_bytes(resp.content)


def _is_stale(path: Path, max_age_days: int = _STALE_DAYS) -> bool:
    if not path.exists():
        return True
    age = (pd.Timestamp.now() - pd.Timestamp(path.stat().st_mtime, unit="s")).days
    return age >= max_age_days


def fetch_epu_us(
    cache: bool = True,
    cache_dir: Path = EPU_CACHE,
) -> pd.DataFrame:
    """Download US EPU Excel.

    Returns DataFrame with month-start DatetimeIndex and columns including:
      EPU_US (3-component index), plus sub-components if available.
    Publication lag: 1 month — caller must apply via pipeline.apply_publication_lags.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    xlsx_path = cache_dir / "epu_us.xlsx"
    parquet_path = cache_dir / "epu_us.parquet"

    if cache and not _is_stale(parquet_path):
        logger.debug("EPU US cache hit")
        return pd.read_parquet(parquet_path)

    if _is_stale(xlsx_path):
        logger.info("Downloading US EPU Excel")
        _download_excel(EPU_US_URL, xlsx_path)

    df = _parse_epu_excel(xlsx_path, primary_col_name="EPU_US")

    if cache:
        df.to_parquet(parquet_path)
    return df


def fetch_epu_global(
    cache: bool = True,
    cache_dir: Path = EPU_CACHE,
) -> pd.DataFrame:
    """Download Global EPU Excel.

    Returns DataFrame with columns: EPU_GLOBAL (GEPU_current), EPU_GLOBAL_PPP.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    xlsx_path = cache_dir / "epu_global.xlsx"
    parquet_path = cache_dir / "epu_global.parquet"

    if cache and not _is_stale(parquet_path):
        logger.debug("EPU Global cache hit")
        return pd.read_parquet(parquet_path)

    if _is_stale(xlsx_path):
        logger.info("Downloading Global EPU Excel")
        _download_excel(EPU_GLOBAL_URL, xlsx_path)

    df = _parse_epu_excel(xlsx_path, primary_col_name="EPU_GLOBAL",
                          alt_names=["gepu_current", "gepu"])

    if cache:
        df.to_parquet(parquet_path)
    return df


def _parse_epu_excel(
    path: Path,
    primary_col_name: str,
    alt_names: list[str] | None = None,
) -> pd.DataFrame:
    """Parse an EPU Excel file into a clean DataFrame with month-start index."""
    sheets = pd.read_excel(path, sheet_name=None, nrows=0, engine="openpyxl")
    # Prefer sheets with longer history (legacy/historical sheets come later)
    sheet_names = list(sheets.keys())
    sheet_name = sheet_names[-1] if len(sheet_names) > 1 else sheet_names[0]
    raw = pd.read_excel(path, sheet_name=sheet_name, engine="openpyxl")

    # Normalise column names
    raw.columns = [str(c).lower().strip() for c in raw.columns]
    cols = list(raw.columns)

    # Build DatetimeIndex from Year/Month integer columns
    year_col  = _find_col(cols, ["year"])
    month_col = _find_col(cols, ["month"])

    if year_col and month_col:
        # Use to_numeric to silently drop attribution text rows at end of file
        year_vals  = pd.to_numeric(raw[year_col],  errors="coerce")
        month_vals = pd.to_numeric(raw[month_col], errors="coerce")
        valid = year_vals.notna() & month_vals.notna()
        raw = raw[valid].copy()
        index = pd.to_datetime(
            {"year":  year_vals[valid].astype(int),
             "month": month_vals[valid].astype(int),
             "day":   1}
        )
    else:
        # Fallback: look for a single date column
        date_col = _find_col(cols, ["date"])
        index = pd.to_datetime(raw[date_col])

    # Find the primary EPU column
    search_names = (alt_names or []) + [primary_col_name.lower(), "three_component_index",
                                         "news_based_policy_uncert_index", "index"]
    primary_col = _find_col(cols, search_names)

    result = pd.DataFrame(index=index)
    result.index.name = "date"

    if primary_col:
        result[primary_col_name] = pd.to_numeric(raw[primary_col].values, errors="coerce")

    # Carry sub-components if present
    for col in cols:
        if col not in (year_col, month_col, primary_col) and raw[col].dtype in (float, int,
                                                                                  "float64", "int64"):
            result[col.upper()] = pd.to_numeric(raw[col].values, errors="coerce")

    result = result.sort_index().dropna(how="all")
    result = result[~result.index.duplicated(keep="first")]
    return result


def _find_col(cols: list[str], candidates: list[str]) -> str | None:
    for col in cols:
        for cand in candidates:
            if cand in col:
                return col
    return None
