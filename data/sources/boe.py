"""
Bank of England Interactive Database (IADB) data downloader.

Provides free historical data via public CSV export.  No API key required.

Primary use: XUDLGPD — daily gold price in USD per troy oz (London PM fix),
available from 1979-01-02 to 2017-05-26.  Automatically spliced with Yahoo
GC=F (COMEX front-month) for the post-2017 period so GLD returns a single
continuous series.

Correlation in overlap 2000–2017: r=0.9999.  Mean price diff ≈ $3.80 (< 0.5%);
disappears in log-return space.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Optional

import pandas as pd
from loguru import logger

BOE_CACHE = Path("data/cache/boe")
CACHE_MAX_AGE_DAYS = 30
_BOE_IADB_URL = "https://www.bankofengland.co.uk/boeapps/database/_iadb-FromShowColumns.asp"
_HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64)",
    "Accept": "text/csv,*/*",
    "Referer": "https://www.bankofengland.co.uk/",
}


def fetch_boe_series(
    series_code: str,
    start_date: str = "1979-01-01",
    end_date: Optional[str] = None,
    cache: bool = True,
    cache_dir: Path = BOE_CACHE,
) -> pd.Series:
    """Download a single BoE IADB series.  Returns pd.Series with DatetimeIndex.

    Caches full history for CACHE_MAX_AGE_DAYS days.  start_date/end_date are
    applied as a slice on the cached data (no re-download needed for subsets).
    """
    import urllib.request

    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{series_code}.parquet"

    if cache and cache_path.exists():
        age_days = (pd.Timestamp.now() - pd.Timestamp(cache_path.stat().st_mtime, unit="s")).days
        if age_days < CACHE_MAX_AGE_DAYS:
            logger.debug(f"BoE cache hit: {series_code}")
            s = pd.read_parquet(cache_path).iloc[:, 0]
            s.name = series_code
            return s.loc[start_date:end_date] if start_date else s

    # BoE IADB CSV export (dayfirst format: DD/Mon/YYYY)
    from_day = pd.Timestamp(start_date or "1970-01-01").strftime("%-d/%-b/%Y") if start_date else "1/Jan/1970"
    to_day   = pd.Timestamp(end_date or pd.Timestamp.today()).strftime("%-d/%-b/%Y") if end_date else pd.Timestamp.today().strftime("%-d/%-b/%Y")

    # Fetch full available history for caching efficiency
    url = (
        f"{_BOE_IADB_URL}?csv.x=yes"
        f"&Datefrom=1/Jan/1970&Dateto={to_day}"
        f"&SeriesCodes={series_code}&CSVF=TT&UsingCodes=Y"
    )
    logger.info(f"Downloading BoE IADB series: {series_code}")
    req = urllib.request.Request(url, headers=_HEADERS)
    with urllib.request.urlopen(req, timeout=30) as r:
        content = r.read().decode(errors="replace")

    lines = content.strip().split("\r\n")
    header_idx = next((i for i, l in enumerate(lines) if l.startswith("DATE,")), None)
    if header_idx is None or header_idx >= len(lines) - 1:
        raise RuntimeError(f"BoE IADB returned no data rows for {series_code}. Response: {lines[:5]}")

    df = pd.read_csv(io.StringIO("\n".join(lines[header_idx:])))
    df["DATE"] = pd.to_datetime(df["DATE"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["DATE"]).set_index("DATE").sort_index()
    df[series_code] = pd.to_numeric(df[series_code], errors="coerce")
    s = df[series_code].dropna()
    s.name = series_code
    logger.info(f"  {series_code}: {s.index[0].date()} → {s.index[-1].date()} ({len(s)} rows)")

    if cache:
        s.to_frame().to_parquet(cache_path)

    return s.loc[start_date:end_date] if start_date else s


def fetch_spliced_gold(
    start_date: str = "1979-01-01",
    end_date: Optional[str] = None,
    series_name: str = "GLD",
    cache: bool = True,
) -> pd.Series:
    """Return a continuous daily gold USD series: BoE XUDLGPD spliced with Yahoo GC=F.

    BoE XUDLGPD (London PM fix) runs 1979-01-02 → 2017-05-26.
    Yahoo GC=F (COMEX front-month) starts 2000-08-30 and is current.

    Splice strategy:
      - Use BoE wherever BoE has data.
      - Fill any remaining gaps (post-2017) with Yahoo GC=F.
    """
    # BoE segment
    try:
        boe_s = fetch_boe_series("XUDLGPD", start_date=start_date, end_date=end_date, cache=cache)
    except Exception as exc:
        logger.warning(f"BoE XUDLGPD download failed: {exc} — falling back to Yahoo GC=F only")
        boe_s = pd.Series(dtype=float, name="XUDLGPD")

    # Yahoo GC=F segment (always fetched to cover post-2017)
    try:
        import yfinance as yf
        gcf = yf.download("GC=F", start=start_date, end=end_date, progress=False, auto_adjust=True)
        if gcf.empty:
            logger.warning("Yahoo GC=F returned no data")
            gcf_s = pd.Series(dtype=float, name="GC=F")
        else:
            gcf_s = gcf["Close"].squeeze()
            gcf_s.index = pd.to_datetime(gcf_s.index)
            gcf_s.name = "GC=F"
    except Exception as exc:
        logger.warning(f"Yahoo GC=F download failed: {exc}")
        gcf_s = pd.Series(dtype=float, name="GC=F")

    # Splice: BoE has priority; GC=F fills gaps (especially post-2017)
    combined = pd.concat([boe_s.rename(series_name), gcf_s.rename(series_name)], axis=1)
    # First column (BoE) wins; second (GC=F) fills NaN
    spliced = combined.iloc[:, 0].combine_first(combined.iloc[:, 1])
    spliced.name = series_name
    spliced = spliced.sort_index()
    spliced = spliced[~spliced.index.duplicated(keep="first")]

    if not spliced.empty:
        logger.info(
            f"Gold (spliced BoE+GC=F): {spliced.dropna().index[0].date()} → "
            f"{spliced.dropna().index[-1].date()} ({spliced.notna().sum()} rows)"
        )

    return spliced
