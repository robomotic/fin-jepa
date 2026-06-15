"""
GPR Index downloader — Caldara & Iacoviello (2022).

Monthly:  https://www.matteoiacoviello.com/gpr_files/gpr_web_latest.xlsx
Daily:    https://www.matteoiacoviello.com/gpr_files/gpr_daily_recent.xlsx

Sub-indices:
  GPR      — overall geopolitical risk
  GPRA     — Acts component (realized events)
  GPRT     — Threats component (threatening language)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import requests
from loguru import logger

GPR_MONTHLY_URL = "https://www.matteoiacoviello.com/gpr_files/data_gpr_export.xls"
GPR_DAILY_URL   = "https://www.matteoiacoviello.com/gpr_files/data_gpr_daily_recent.xls"
GPR_CACHE = Path("data/cache/gpr")

# How many days before re-downloading (index updates monthly)
_STALE_DAYS = 30


def _download_excel(url: str, dest: Path) -> None:
    headers = {"User-Agent": "Mozilla/5.0 fin-jepa research project"}
    resp = requests.get(url, headers=headers, timeout=60)
    resp.raise_for_status()
    dest.write_bytes(resp.content)


def _is_stale(path: Path, max_age_days: int = _STALE_DAYS) -> bool:
    if not path.exists():
        return True
    age = (pd.Timestamp.now() - pd.Timestamp(path.stat().st_mtime, unit="s")).days
    return age >= max_age_days


def fetch_gpr_monthly(
    cache: bool = True,
    cache_dir: Path = GPR_CACHE,
) -> pd.DataFrame:
    """Download and parse monthly GPR Excel.

    Returns DataFrame with DatetimeIndex (month-start) and columns:
      GPR_GLOBAL, GPRA, GPRT
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    xlsx_path = cache_dir / "gpr_monthly.xls"
    parquet_path = cache_dir / "gpr_monthly.parquet"

    if cache and not _is_stale(parquet_path):
        logger.debug("GPR monthly cache hit")
        return pd.read_parquet(parquet_path)

    if _is_stale(xlsx_path):
        logger.info("Downloading GPR monthly Excel")
        _download_excel(GPR_MONTHLY_URL, xlsx_path)

    df = _parse_gpr_excel(xlsx_path, frequency="monthly")

    if cache:
        df.to_parquet(parquet_path)
    return df


def fetch_gpr_daily(
    start_date: str = "2000-01-01",
    cache: bool = True,
    cache_dir: Path = GPR_CACHE,
) -> pd.DataFrame:
    """Download and parse daily GPR Excel.

    Returns DataFrame with DatetimeIndex and columns:
      GPR_GLOBAL, GPRA, GPRT
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    xlsx_path = cache_dir / "gpr_daily.xls"
    parquet_path = cache_dir / "gpr_daily.parquet"

    if cache and not _is_stale(parquet_path):
        logger.debug("GPR daily cache hit")
        df = pd.read_parquet(parquet_path)
        return df.loc[start_date:]

    if _is_stale(xlsx_path):
        logger.info("Downloading GPR daily Excel")
        _download_excel(GPR_DAILY_URL, xlsx_path)

    df = _parse_gpr_excel(xlsx_path, frequency="daily")

    if cache:
        df.to_parquet(parquet_path)

    return df.loc[start_date:]


def _parse_gpr_excel(path: Path, frequency: str) -> pd.DataFrame:
    """Parse raw GPR Excel into a clean DataFrame.

    The file layout can change with new vintages. We inspect column names
    and build DatetimeIndex from year/month (monthly) or date (daily) columns.
    """
    engine = "xlrd" if str(path).endswith(".xls") else "openpyxl"
    # Inspect available sheets
    sheets = pd.read_excel(path, sheet_name=None, nrows=0, engine=engine)
    sheet_name = list(sheets.keys())[0]
    logger.debug(f"GPR Excel: using sheet '{sheet_name}' from {path.name}")

    raw = pd.read_excel(path, sheet_name=sheet_name, engine=engine)
    cols_lower = {c: c.lower().strip() for c in raw.columns}
    raw.rename(columns=cols_lower, inplace=True)
    cols = list(raw.columns)

    # Build DatetimeIndex
    if frequency == "monthly":
        # Expected: columns 'year' and 'month' (integer)
        year_col = _find_col(cols, ["year"])
        month_col = _find_col(cols, ["month"])
        index = pd.to_datetime(
            {"year": raw[year_col].astype(int),
             "month": raw[month_col].astype(int),
             "day": 1}
        )
    else:
        # Daily: look for a date-like column
        date_col = _find_col(cols, ["date", "day", "time"])
        series = raw[date_col]
        # YYYYMMDD integer format (e.g. 19850101) — must be cast to string first
        if pd.api.types.is_numeric_dtype(series):
            index = pd.to_datetime(series.astype(int).astype(str), format="%Y%m%d", errors="coerce")
        else:
            index = pd.to_datetime(series, errors="coerce")

    # Extract GPR columns
    gpr_col  = _find_col(cols, ["gprd", "gpr", "gprglobal", "gpr_global", "gprnews"])
    gpra_col = _find_col(cols, ["gprd_act", "gpra", "gpr_act", "gpr_acts"])
    gprt_col = _find_col(cols, ["gprd_threat", "gprt", "gpr_thr", "gpr_threats"])

    result = pd.DataFrame(index=index)
    result.index.name = "date"

    if gpr_col:
        result["GPR_GLOBAL"] = raw[gpr_col].values
    if gpra_col:
        result["GPRA"] = raw[gpra_col].values
    if gprt_col:
        result["GPRT"] = raw[gprt_col].values

    result = result.sort_index()
    result = result[~result.index.duplicated(keep="first")]
    return result


def _find_col(cols: list[str], candidates: list[str]) -> Optional[str]:
    """Find the first column name that contains any candidate substring."""
    for col in cols:
        for cand in candidates:
            if cand in col:
                return col
    return None
