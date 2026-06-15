"""
Yahoo Finance downloader via yfinance.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import yfinance as yf
from loguru import logger

YAHOO_CACHE = Path("data/cache/yahoo")

# Map our canonical names to Yahoo Finance symbols where they differ
_SYMBOL_MAP: dict[str, str] = {
    "DXY": "DX-Y.NYB",
    "VIX": "^VIX",
    "BDI": "^BDI",
    "MOVE": "^MOVE",
}


def _to_yahoo_symbol(name: str, cfg: dict) -> str:
    """Resolve canonical name to Yahoo ticker symbol."""
    return cfg.get("id", name)


def fetch_ticker(
    name: str,
    symbol: str,
    start_date: str = "1990-01-01",
    end_date: Optional[str] = None,
    cache: bool = True,
    cache_dir: Path = YAHOO_CACHE,
) -> pd.Series:
    """Download Adj Close for a single ticker. Returns pd.Series."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{name}.parquet"

    if cache and cache_path.exists():
        age_days = (pd.Timestamp.now() - pd.Timestamp(cache_path.stat().st_mtime, unit="s")).days
        if age_days < 1:
            logger.debug(f"Yahoo cache hit: {name}")
            df = pd.read_parquet(cache_path)
            s = df.iloc[:, 0]
            s.name = name
            return s.loc[start_date:]

    logger.info(f"Downloading Yahoo ticker: {symbol} (as {name})")
    ticker = yf.Ticker(symbol)
    hist = ticker.history(start=start_date, end=end_date, auto_adjust=True)

    if hist.empty:
        logger.warning(f"No data returned for {symbol}")
        return pd.Series(name=name, dtype=float)

    s = hist["Close"].rename(name)
    s.index = pd.to_datetime(s.index).tz_localize(None)

    if cache:
        s.to_frame().to_parquet(cache_path)

    return s


def fetch_all_yahoo_tickers(
    series_map: dict[str, str],  # {our_name: yahoo_symbol}
    start_date: str = "1990-01-01",
    end_date: Optional[str] = None,
    cache: bool = True,
    cache_dir: Path = YAHOO_CACHE,
) -> pd.DataFrame:
    """Batch download. Returns DataFrame of Adj Close prices (not returns)."""
    frames: dict[str, pd.Series] = {}
    for name, symbol in series_map.items():
        try:
            s = fetch_ticker(name, symbol, start_date=start_date, end_date=end_date,
                             cache=cache, cache_dir=cache_dir)
            if not s.empty:
                frames[name] = s
        except Exception as exc:
            logger.warning(f"Failed to download {name} ({symbol}): {exc}")

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames.values(), axis=1).sort_index()


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute log returns: log(P_t / P_{t-1}). First row will be NaN."""
    import numpy as np
    return np.log(prices / prices.shift(1))


def build_yahoo_series_map(config: dict) -> dict[str, str]:
    """Extract {our_name: yahoo_symbol} for all yahoo-sourced series."""
    return {
        name: cfg["id"]
        for name, cfg in config["series"].items()
        if cfg["source"] == "yahoo"
    }
