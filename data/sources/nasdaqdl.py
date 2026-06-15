"""
Nasdaq Data Link (formerly Quandl) data downloader.
Requires: nasdaq-data-link (pip install nasdaq-data-link)

API key setup (free registration at data.nasdaq.com):
  1. export NASDAQ_DATA_LINK_API_KEY=your_key_here
  2. or: echo "your_key_here" > ~/.nasdaq_data_link_api_key

Provides:
  - LBMA/GOLD  : London Bullion Market Association gold daily fix (USD, from 1968)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import pandas as pd
from loguru import logger

NASDAQDL_CACHE = Path("data/cache/nasdaqdl")
CACHE_MAX_AGE_DAYS = 30


def _get_api_key() -> Optional[str]:
    key = os.environ.get("NASDAQ_DATA_LINK_API_KEY")
    if not key:
        key_file = Path.home() / ".nasdaq_data_link_api_key"
        if key_file.exists():
            key = key_file.read_text().strip()
    return key


def fetch_dataset(
    dataset_code: str,
    column: str,
    series_name: str,
    start_date: str = "1960-01-01",
    end_date: Optional[str] = None,
    cache: bool = True,
    cache_dir: Path = NASDAQDL_CACHE,
) -> pd.Series:
    """Download a single column from a Nasdaq Data Link dataset.

    Returns pd.Series with DatetimeIndex, named series_name.
    Caches to parquet for CACHE_MAX_AGE_DAYS days.

    Args:
        dataset_code: e.g. "LBMA/GOLD"
        column:       column name in the returned DataFrame, e.g. "USD (PM)"
        series_name:  name for the output series, e.g. "GLD"
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    safe_name = dataset_code.replace("/", "_")
    cache_path = cache_dir / f"{safe_name}.parquet"

    if cache and cache_path.exists():
        age_days = (pd.Timestamp.now() - pd.Timestamp(cache_path.stat().st_mtime, unit="s")).days
        if age_days < CACHE_MAX_AGE_DAYS:
            logger.debug(f"Nasdaq DL cache hit: {dataset_code}")
            df = pd.read_parquet(cache_path)
            if column in df.columns:
                s = df[column].rename(series_name)
                return s.loc[start_date:] if start_date else s

    api_key = _get_api_key()
    if not api_key:
        logger.warning(
            f"NASDAQ_DATA_LINK_API_KEY not set — skipping {dataset_code}. "
            "Register free at data.nasdaq.com and set the key in .env or "
            "~/.nasdaq_data_link_api_key"
        )
        return pd.Series(dtype=float, name=series_name)

    try:
        import nasdaqdatalink
    except ImportError:
        logger.error("nasdaq-data-link not installed. Run: pip install nasdaq-data-link")
        return pd.Series(dtype=float, name=series_name)

    nasdaqdatalink.ApiConfig.api_key = api_key
    logger.info(f"Downloading Nasdaq DL dataset: {dataset_code}")

    try:
        df = nasdaqdatalink.get(dataset_code, start_date=start_date, end_date=end_date)
    except Exception as exc:
        logger.error(f"Nasdaq DL download failed for {dataset_code}: {exc}")
        return pd.Series(dtype=float, name=series_name)

    if column not in df.columns:
        available = list(df.columns)
        logger.error(f"{dataset_code}: column '{column}' not found. Available: {available}")
        return pd.Series(dtype=float, name=series_name)

    if cache:
        df.to_parquet(cache_path)

    s = df[column].rename(series_name)
    s.index = pd.to_datetime(s.index)
    return s.loc[start_date:] if start_date else s


def fetch_lbma_gold(
    start_date: str = "1968-01-01",
    end_date: Optional[str] = None,
    fix: str = "PM",
) -> pd.Series:
    """Fetch LBMA gold daily fix price in USD.

    Args:
        fix: "AM" or "PM" (London morning / afternoon fix). Default PM.
    Returns:
        pd.Series of gold prices (USD/troy oz) with DatetimeIndex.
    """
    column = f"USD ({fix})"
    return fetch_dataset(
        dataset_code="LBMA/GOLD",
        column=column,
        series_name="GLD",
        start_date=start_date,
        end_date=end_date,
    )
