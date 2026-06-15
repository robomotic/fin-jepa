"""
Data pipeline: load all sources → apply publication lags → harmonize to
business days → transform → expanding z-score → embargo splits.

Usage:
    from data.pipeline import build_pipeline
    splits = build_pipeline()   # {'train': df, 'val': df, 'test': df}
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml
from loguru import logger


# ─── Config ───────────────────────────────────────────────────────────────────

def load_config(config_path: str = "config/variables.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


# ─── Raw Panel ────────────────────────────────────────────────────────────────

def build_raw_panel(config: dict, start_date: str = "1999-01-01") -> pd.DataFrame:
    """Download all series defined in config['series'].

    Returns a wide DataFrame with a DatetimeIndex (calendar daily).
    Values are raw prices / levels — NOT yet transformed or normalised.
    """
    from data.sources.fred import fetch_all_fred_series, build_fred_series_map
    from data.sources.yahoo import fetch_all_yahoo_tickers, build_yahoo_series_map
    from data.sources.gpr import fetch_gpr_daily
    from data.sources.epu import fetch_epu_us, fetch_epu_global
    from data.sources.gscpi import fetch_gscpi

    frames: list[pd.DataFrame] = []

    # FRED
    fred_map = build_fred_series_map(config)
    if fred_map:
        fred_df = fetch_all_fred_series(fred_map, start_date=start_date)
        frames.append(fred_df)

    # Yahoo
    yahoo_map = build_yahoo_series_map(config)
    if yahoo_map:
        yahoo_df = fetch_all_yahoo_tickers(yahoo_map, start_date=start_date)
        frames.append(yahoo_df)

    # GPR
    gpr_series = {n: c for n, c in config["series"].items() if c["source"] == "gpr"}
    if gpr_series:
        try:
            gpr_df = fetch_gpr_daily(start_date=start_date)
            # Map internal GPR column names to our canonical names
            col_map = {"GPR_GLOBAL": "GPR_GLOBAL", "GPRA": "GPRA", "GPRT": "GPRT"}
            gpr_df = gpr_df.rename(columns={v: k for k, v in col_map.items() if v in gpr_df.columns})
            frames.append(gpr_df)
        except Exception as e:
            logger.warning(f"GPR download failed: {e}")

    # EPU
    epu_series = {n: c for n, c in config["series"].items() if c["source"] == "epu"}
    if epu_series:
        try:
            epu_us = fetch_epu_us()
            if "EPU_US" in epu_us.columns:
                frames.append(epu_us[["EPU_US"]])
            epu_gl = fetch_epu_global()
            if "EPU_GLOBAL" in epu_gl.columns:
                frames.append(epu_gl[["EPU_GLOBAL"]])
        except Exception as e:
            logger.warning(f"EPU download failed: {e}")

    # GSCPI
    if any(c["source"] == "gscpi" for c in config["series"].values()):
        try:
            gscpi = fetch_gscpi()
            frames.append(gscpi.to_frame())
        except Exception as e:
            logger.warning(f"GSCPI download failed: {e}")

    if not frames:
        raise RuntimeError("No data downloaded. Check API keys and network.")

    panel = pd.concat(frames, axis=1)
    panel.index = pd.to_datetime(panel.index)
    panel = panel.sort_index()
    panel = panel[~panel.index.duplicated(keep="last")]
    return panel


# ─── Publication Lags ─────────────────────────────────────────────────────────

def apply_publication_lags(panel: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Shift lagged series forward by pub_lag_days calendar days.

    This ensures e.g. January CPI (published mid-February) only appears
    in the dataset from mid-February onwards, preventing look-ahead.

    Must be called BEFORE harmonize_to_business_days.
    """
    series_cfg = config.get("series", {})
    result = panel.copy()

    for col in panel.columns:
        cfg = series_cfg.get(col, {})
        lag_days = cfg.get("pub_lag_days", 0)
        if lag_days and lag_days > 0:
            shifted = result[col].copy()
            # Shift the index forward: value at date D becomes available at D + lag
            shifted.index = shifted.index + pd.DateOffset(days=lag_days)
            result[col] = shifted
            logger.debug(f"Applied {lag_days}d publication lag to {col}")

    return result


# ─── Business Day Harmonisation ───────────────────────────────────────────────

def harmonize_to_business_days(
    panel: pd.DataFrame,
    start_date: str,
    end_date: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reindex to NYSE business days, forward-fill gaps.

    Returns:
        (harmonized_panel, ffill_mask)
        ffill_mask is True where a value was forward-filled.
    """
    try:
        import pandas_market_calendars as mcal
        nyse = mcal.get_calendar("NYSE")
        schedule = nyse.schedule(start_date=start_date, end_date=end_date)
        bdays = mcal.date_range(schedule, frequency="1D").normalize().tz_localize(None)
        bdays = pd.DatetimeIndex(bdays.date).rename(None)
    except Exception:
        logger.warning("pandas_market_calendars not available; falling back to pandas BDay")
        bdays = pd.bdate_range(start=start_date, end=end_date)

    original_mask = panel.notna()
    harmonized = panel.reindex(bdays, method=None)  # reindex without fill first
    harmonized = harmonized.ffill()
    # ffill_mask: True where we filled (was NaN before, not NaN after)
    was_na_before = ~original_mask.reindex(bdays, fill_value=False)
    is_filled_now  = harmonized.notna()
    ffill_mask = was_na_before & is_filled_now

    return harmonized, ffill_mask


# ─── Transforms ───────────────────────────────────────────────────────────────

def apply_transforms(panel: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Apply per-series transforms defined in variables.yaml.

    Transforms:
      log_return — log(P_t / P_{t-1})
      diff       — P_t - P_{t-1}
      level      — no change
    """
    series_cfg = config.get("series", {})
    result = pd.DataFrame(index=panel.index)

    for col in panel.columns:
        cfg = series_cfg.get(col, {})
        transform = cfg.get("transform", "level")
        s = panel[col]

        if transform == "log_return":
            result[col] = np.log(s / s.shift(1)).replace([np.inf, -np.inf], np.nan)
        elif transform == "diff":
            result[col] = s.diff()
        else:
            result[col] = s

    return result


# ─── Expanding Z-score ────────────────────────────────────────────────────────

def compute_expanding_zscore(
    panel: pd.DataFrame,
    min_periods: int = 252,
    clip_sigma: float = 5.0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Z-score each column using expanding-window mean and std.

    NaN is left during the burn-in period (first min_periods rows).
    Clipping at ±clip_sigma reduces outlier influence.

    Returns:
        (z_panel, expanding_means, expanding_stds)
    """
    means = panel.expanding(min_periods=min_periods).mean()
    stds  = panel.expanding(min_periods=min_periods).std()

    z = (panel - means) / stds.replace(0, np.nan)
    z = z.clip(lower=-clip_sigma, upper=clip_sigma)

    return z, means, stds


# ─── Missing Window Flags ─────────────────────────────────────────────────────

def flag_missing_windows(
    panel: pd.DataFrame,
    window_size: int,
    threshold: float = 0.20,
) -> pd.Series:
    """For each possible window start, flag if > threshold fraction of cells are NaN.

    Returns pd.Series[bool] aligned to panel index.
    True = window starting here is INVALID (too many missing values).
    """
    total_cells = window_size * panel.shape[1]
    rolling_nans = panel.isna().rolling(window=window_size, min_periods=1).sum().sum(axis=1)
    invalid = rolling_nans / total_cells > threshold
    return invalid


# ─── Train / Val / Test Splits ────────────────────────────────────────────────

def make_train_val_test_splits(
    panel: pd.DataFrame,
    config: dict,
) -> dict[str, pd.DataFrame]:
    """Apply walk-forward splits with embargo from config['splits'].

    The embargo_days business days between train_end and val_start
    are excluded entirely (not assigned to either set).
    """
    sp = config["splits"]
    train_end   = pd.Timestamp(sp["train_end"])
    embargo     = sp.get("embargo_days", 20)
    val_start   = pd.Timestamp(sp["val_start"])
    val_end     = pd.Timestamp(sp["val_end"])
    test_start  = pd.Timestamp(sp["test_start"])
    test_end    = pd.Timestamp(sp["test_end"])
    train_start = pd.Timestamp(sp["train_start"])

    # Verify embargo gap
    gap_days = (val_start - train_end).days
    if gap_days < embargo:
        logger.warning(
            f"val_start is only {gap_days} calendar days after train_end "
            f"(configured embargo={embargo} business days). "
            f"Check config splits."
        )

    train = panel.loc[train_start:train_end]
    val   = panel.loc[val_start:val_end]
    test  = panel.loc[test_start:test_end]

    logger.info(f"Train: {train.index[0].date()} → {train.index[-1].date()} ({len(train)} rows)")
    logger.info(f"Val:   {val.index[0].date()} → {val.index[-1].date()} ({len(val)} rows)")
    logger.info(f"Test:  {test.index[0].date()} → {test.index[-1].date()} ({len(test)} rows)")

    return {"train": train, "val": val, "test": test}


# ─── Master Build ─────────────────────────────────────────────────────────────

def build_pipeline(
    config_path: str = "config/variables.yaml",
    force_rebuild: bool = False,
    run_diagnostics: bool = True,
    save_splits: bool = True,
) -> dict[str, pd.DataFrame]:
    """End-to-end pipeline. Returns {'train': df, 'val': df, 'test': df}.

    Step order (order matters for correctness):
      1. Download raw data
      2. Apply publication lags   ← must come BEFORE forward-fill
      3. Harmonize to business days + forward-fill
      4. Apply per-series transforms (log_return, diff, level)
      5. Run diagnostics (optional)
      6. Expanding z-score normalisation
      7. Embargo splits
    """
    splits_cache = Path("data/cache/splits")
    if not force_rebuild and splits_cache.exists() and any(splits_cache.iterdir()):
        logger.info("Loading cached splits from data/cache/splits/")
        return {
            split: pd.read_parquet(splits_cache / f"{split}.parquet")
            for split in ("train", "val", "test")
            if (splits_cache / f"{split}.parquet").exists()
        }

    config = load_config(config_path)
    sp = config["splits"]

    # 1. Download
    logger.info("Step 1/6: Downloading raw data")
    panel = build_raw_panel(config, start_date="1991-01-01")
    logger.info(f"Raw panel: {panel.shape} ({panel.columns.tolist()})")

    # 2. Publication lags
    logger.info("Step 2/6: Applying publication lags")
    panel = apply_publication_lags(panel, config)

    # 3. Harmonise to business days
    logger.info("Step 3/6: Harmonizing to NYSE business days")
    panel, ffill_mask = harmonize_to_business_days(
        panel, start_date=sp["train_start"], end_date=sp["test_end"]
    )

    # 4. Transforms
    logger.info("Step 4/6: Applying per-series transforms")
    panel = apply_transforms(panel, config)

    # 5. Diagnostics (on transformed panel)
    if run_diagnostics:
        logger.info("Step 5/6: Running statistical diagnostics")
        try:
            from data.diagnostics import build_diagnostics_report
            build_diagnostics_report(panel, config, save_figures=True)
        except Exception as e:
            logger.warning(f"Diagnostics failed (non-fatal): {e}")

    # 6. Z-score
    logger.info("Step 6/6: Expanding z-score normalisation")
    norm_cfg = config.get("normalization", {})
    z_panel, _, _ = compute_expanding_zscore(
        panel,
        min_periods=norm_cfg.get("min_burn_in_days", 252),
        clip_sigma=norm_cfg.get("clip_sigma", 5.0),
    )

    # 7. Splits
    splits = make_train_val_test_splits(z_panel, config)

    # Cache
    if save_splits:
        splits_cache.mkdir(parents=True, exist_ok=True)
        for name, df in splits.items():
            df.to_parquet(splits_cache / f"{name}.parquet")
        logger.info(f"Splits cached to {splits_cache}")

    return splits
