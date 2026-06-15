"""
Automated statistical diagnostics for all series in the panel.

Runs once at pipeline build time. Saves:
  data/cache/diagnostics/diagnostics_report.json
  data/cache/diagnostics/figures/acf_<series>.png
  data/cache/diagnostics/figures/stl_<series>.png

Key outputs:
  - ADF + KPSS stationarity verdict per series
  - STL decomposition (trend / seasonal / residual)
  - ACF / PACF max significant lag → informs minimum context window
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

DIAG_DIR = Path("data/cache/diagnostics")
FIGURE_DIR = DIAG_DIR / "figures"

# Seasonal period by native frequency
_SEASONAL_PERIOD = {
    "daily":   252,  # trading year
    "weekly":  52,
    "monthly": 12,
}


# ─── Stationarity ─────────────────────────────────────────────────────────────

def run_stationarity_tests(series: pd.Series, alpha: float = 0.05) -> dict:
    """Run ADF and KPSS on a series.

    ADF  null hypothesis = unit root  → stationary if p < alpha
    KPSS null hypothesis = stationary → stationary if p > alpha
    Both must agree for verdict == "stationary".
    """
    from statsmodels.tsa.stattools import adfuller, kpss

    clean = series.dropna()
    if len(clean) < 20:
        return {"verdict": "insufficient_data", "n_obs": len(clean)}

    # ADF
    adf_stat, adf_p, *_ = adfuller(clean, autolag="AIC")
    adf_stationary = bool(adf_p < alpha)

    # KPSS — suppress the frequency warning
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kpss_stat, kpss_p, *_ = kpss(clean, regression="c", nlags="auto")
    kpss_stationary = bool(kpss_p > alpha)

    if adf_stationary and kpss_stationary:
        verdict = "stationary"
    elif not adf_stationary and not kpss_stationary:
        verdict = "unit_root"
    else:
        verdict = "conflicting"

    return {
        "adf_statistic": float(adf_stat),
        "adf_pvalue": float(adf_p),
        "adf_is_stationary": adf_stationary,
        "kpss_statistic": float(kpss_stat),
        "kpss_pvalue": float(kpss_p),
        "kpss_is_stationary": kpss_stationary,
        "verdict": verdict,
        "n_obs": len(clean),
    }


def auto_transform_to_stationarity(
    series: pd.Series,
    prescribed_transform: str,
    max_extra_diffs: int = 1,
    alpha: float = 0.05,
) -> tuple[pd.Series, str, int]:
    """Apply prescribed transform, then check stationarity.

    If still non-stationary, applies one additional diff and warns.
    Does NOT override variables.yaml — config is authoritative.

    Returns:
        (transformed_series, description, extra_diffs_applied)
    """
    s = _apply_transform(series, prescribed_transform)
    result = run_stationarity_tests(s, alpha=alpha)
    extra = 0

    if result["verdict"] == "unit_root" and extra < max_extra_diffs:
        logger.warning(
            f"{series.name}: non-stationary after '{prescribed_transform}'. "
            f"Applying 1 additional diff. Consider updating variables.yaml."
        )
        s = s.diff()
        extra = 1

    return s, prescribed_transform + ("+diff" if extra else ""), extra


def _apply_transform(series: pd.Series, transform: str) -> pd.Series:
    if transform == "log_return":
        return np.log(series / series.shift(1)).replace([np.inf, -np.inf], np.nan)
    if transform == "diff":
        return series.diff()
    # "level" — return as-is
    return series.copy()


# ─── STL Decomposition ────────────────────────────────────────────────────────

def stl_decompose(
    series: pd.Series,
    frequency: str = "daily",
    robust: bool = True,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """STL decomposition. Returns (trend, seasonal, residual).

    Useful as an ablation: does JEPA perform better on de-trended residuals?
    """
    from statsmodels.tsa.seasonal import STL

    clean = series.dropna()
    period = _SEASONAL_PERIOD.get(frequency, 252)

    if len(clean) < 2 * period:
        logger.warning(f"{series.name}: too short for STL (need {2*period}, got {len(clean)})")
        nans = pd.Series(np.nan, index=clean.index, name=series.name)
        return nans, nans, clean

    stl = STL(clean, period=period, robust=robust)
    result = stl.fit()

    trend    = pd.Series(result.trend,    index=clean.index, name=series.name)
    seasonal = pd.Series(result.seasonal, index=clean.index, name=series.name)
    resid    = pd.Series(result.resid,    index=clean.index, name=series.name)
    return trend, seasonal, resid


# ─── ACF / PACF ───────────────────────────────────────────────────────────────

def compute_acf_profile(
    series: pd.Series,
    max_lag: int = 252,
    alpha: float = 0.05,
) -> dict:
    """ACF and PACF up to max_lag.

    Returns dict with max_significant_acf_lag: the largest lag outside the CI.
    This value sets a floor on the minimum context window for JEPA.
    """
    from statsmodels.tsa.stattools import acf, pacf

    clean = series.dropna()
    n = len(clean)
    if n < max_lag + 10:
        max_lag = max(10, n // 4)

    acf_vals, acf_ci = acf(clean, nlags=max_lag, alpha=alpha, fft=True)
    pacf_vals, pacf_ci = pacf(clean, nlags=min(max_lag, n // 2 - 2), alpha=alpha)

    # CI is symmetric around zero for a white-noise null: 1.96/sqrt(n)
    threshold = 1.96 / np.sqrt(n)

    # Find max significant lag (skip lag-0 which is always 1)
    sig_acf_lags  = [lag for lag, v in enumerate(acf_vals[1:], 1)  if abs(v) > threshold]
    sig_pacf_lags = [lag for lag, v in enumerate(pacf_vals[1:], 1) if abs(v) > threshold]

    return {
        "max_significant_acf_lag":  max(sig_acf_lags)  if sig_acf_lags  else 0,
        "max_significant_pacf_lag": max(sig_pacf_lags) if sig_pacf_lags else 0,
        "n_significant_acf_lags":   len(sig_acf_lags),
        "acf_values":  [float(v) for v in acf_vals[:max_lag + 1]],
        "pacf_values": [float(v) for v in pacf_vals],
        "n_obs": n,
    }


# ─── Full Diagnostics Report ──────────────────────────────────────────────────

def build_diagnostics_report(
    panel: pd.DataFrame,
    config: dict,
    output_dir: Path = DIAG_DIR,
    save_figures: bool = True,
) -> dict:
    """Run all diagnostics on every series in panel.

    Saves diagnostics_report.json and optional matplotlib figures.
    Returns the full report dict.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = output_dir / "figures"
    if save_figures:
        fig_dir.mkdir(parents=True, exist_ok=True)

    series_cfg = config.get("series", {})
    report: dict[str, dict] = {}

    for col in panel.columns:
        cfg = series_cfg.get(col, {})
        frequency = cfg.get("frequency", "daily")
        prescribed = cfg.get("transform", "level")

        logger.info(f"Diagnostics: {col}")
        s = panel[col].dropna()

        if len(s) < 30:
            report[col] = {"verdict": "insufficient_data", "n_obs": len(s)}
            continue

        # Apply prescribed transform before stationarity test
        s_transformed = _apply_transform(s, prescribed)

        stationarity = run_stationarity_tests(s_transformed, alpha=0.05)

        # Extra diff check
        extra_diffs = 0
        if stationarity["verdict"] == "unit_root":
            logger.warning(
                f"{col}: still non-stationary after '{prescribed}'. "
                f"Consider changing transform in variables.yaml."
            )
            extra_diffs = 1

        acf_profile = compute_acf_profile(s_transformed, max_lag=min(252, len(s_transformed) // 4))

        report[col] = {
            "frequency": frequency,
            "prescribed_transform": prescribed,
            "diffs_beyond_prescribed": extra_diffs,
            **stationarity,
            **{f"acf_{k}": v for k, v in acf_profile.items()
               if k not in ("acf_values", "pacf_values")},
        }

        if save_figures:
            _plot_acf(col, acf_profile, fig_dir)
            try:
                _plot_stl(col, s, frequency, fig_dir)
            except Exception as e:
                logger.debug(f"STL plot skipped for {col}: {e}")

    # Console summary
    non_stationary = [k for k, v in report.items() if v.get("verdict") == "unit_root"]
    max_acf_lag = max(
        (v.get("acf_max_significant_acf_lag", 0) for v in report.values()),
        default=0
    )

    if non_stationary:
        logger.warning(f"Non-stationary after prescribed transform: {non_stationary}")
    logger.info(f"Max significant ACF lag across all series: {max_acf_lag} days")

    configured_ctx = config.get("model", {}).get("patch_len", 21) * config.get("model", {}).get("n_patches_context", 9)
    if max_acf_lag > configured_ctx:
        logger.warning(
            f"Recommended context_len ({max_acf_lag}) > configured ({configured_ctx}). "
            f"Consider increasing n_patches_context in variables.yaml."
        )
    else:
        logger.info(f"Configured context_len={configured_ctx} — OK (>= {max_acf_lag})")

    # Save report
    report_path = output_dir / "diagnostics_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info(f"Diagnostics report saved to {report_path}")

    return report


def _plot_acf(name: str, acf_profile: dict, fig_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
        from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    except ImportError:
        return

    fig, axes = plt.subplots(2, 1, figsize=(12, 6))
    acf_vals = acf_profile["acf_values"]
    lags = range(len(acf_vals))

    axes[0].bar(lags, acf_vals, width=0.3)
    axes[0].axhline(0, color="black", linewidth=0.5)
    axes[0].axhline(1.96 / acf_profile["n_obs"] ** 0.5, color="blue", linestyle="--", linewidth=0.8)
    axes[0].axhline(-1.96 / acf_profile["n_obs"] ** 0.5, color="blue", linestyle="--", linewidth=0.8)
    axes[0].set_title(f"ACF — {name}")

    pacf_vals = acf_profile["pacf_values"]
    lags_p = range(len(pacf_vals))
    axes[1].bar(lags_p, pacf_vals, width=0.3)
    axes[1].axhline(0, color="black", linewidth=0.5)
    axes[1].axhline(1.96 / acf_profile["n_obs"] ** 0.5, color="blue", linestyle="--", linewidth=0.8)
    axes[1].axhline(-1.96 / acf_profile["n_obs"] ** 0.5, color="blue", linestyle="--", linewidth=0.8)
    axes[1].set_title(f"PACF — {name}")

    plt.tight_layout()
    fig.savefig(fig_dir / f"acf_{name}.png", dpi=100)
    plt.close(fig)


def _plot_stl(name: str, series: pd.Series, frequency: str, fig_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    trend, seasonal, resid = stl_decompose(series, frequency=frequency)
    if trend.isna().all():
        return

    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    series.plot(ax=axes[0], title=f"Original — {name}", linewidth=0.7)
    trend.plot(ax=axes[1], title="Trend", linewidth=0.7)
    seasonal.plot(ax=axes[2], title="Seasonal", linewidth=0.7)
    resid.plot(ax=axes[3], title="Residual", linewidth=0.7)

    plt.tight_layout()
    fig.savefig(fig_dir / f"stl_{name}.png", dpi=100)
    plt.close(fig)
