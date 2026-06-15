"""
FRED data downloader.
Requires: fredapi, pandas

API key priority:
  1. FRED_API_KEY environment variable
  2. ~/.fred_api_key file
  3. Anonymous (rate-limited to ~120 req/min)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import pandas as pd
from loguru import logger

FRED_CACHE = Path("data/cache/fred")

# NFCI sub-index FRED series IDs
NFCI_SUBINDICES = {
    "NFCI": "NFCI",
    "NFCI_RISK": "NFCIRISK",
    "NFCI_CREDIT": "NFCICREDIT",
    "NFCI_LEVERAGE": "NFCILEVERAGE",
    "STLFSI": "STLFSI4",
}


def _get_client():
    from fredapi import Fred

    api_key = os.environ.get("FRED_API_KEY")
    if not api_key:
        key_file = Path.home() / ".fred_api_key"
        if key_file.exists():
            api_key = key_file.read_text().strip()
    return Fred(api_key=api_key)


def fetch_series(
    series_id: str,
    start_date: str = "1990-01-01",
    end_date: Optional[str] = None,
    cache: bool = True,
    cache_dir: Path = FRED_CACHE,
) -> pd.Series:
    """Download a single FRED series. Returns pd.Series with DatetimeIndex.

    Caches to parquet. Re-downloads if cache is older than 7 days.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{series_id}.parquet"

    if cache and cache_path.exists():
        age_days = (pd.Timestamp.now() - pd.Timestamp(cache_path.stat().st_mtime, unit="s")).days
        if age_days < 7:
            logger.debug(f"FRED cache hit: {series_id}")
            df = pd.read_parquet(cache_path)
            s = df.iloc[:, 0]
            s.name = series_id
            return s.loc[start_date:]

    logger.info(f"Downloading FRED series: {series_id}")
    fred = _get_client()
    s = fred.get_series(series_id, observation_start=start_date, observation_end=end_date)
    s.name = series_id
    s.index = pd.to_datetime(s.index)

    if cache:
        s.to_frame().to_parquet(cache_path)

    return s


def fetch_all_fred_series(
    series_map: dict[str, str],  # {our_name: fred_id}
    start_date: str = "1990-01-01",
    end_date: Optional[str] = None,
    cache: bool = True,
    cache_dir: Path = FRED_CACHE,
) -> pd.DataFrame:
    """Batch fetch. Returns aligned DataFrame (outer join on date index)."""
    frames: dict[str, pd.Series] = {}
    for name, fred_id in series_map.items():
        try:
            s = fetch_series(fred_id, start_date=start_date, end_date=end_date,
                             cache=cache, cache_dir=cache_dir)
            s.name = name
            frames[name] = s
        except Exception as exc:
            logger.warning(f"Failed to download {name} ({fred_id}): {exc}")

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames.values(), axis=1).sort_index()


def build_fred_series_map(config: dict) -> dict[str, str]:
    """Extract {our_name: fred_id} for all fred-sourced series from config."""
    return {
        name: cfg["id"]
        for name, cfg in config["series"].items()
        if cfg["source"] == "fred"
    }
