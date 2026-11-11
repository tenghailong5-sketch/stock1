"""Optional CNEquity adapter with strict schema checks.

CNEquity is a candidate source for persisted A-share history. This adapter
never upgrades retrieval timestamps into historical point-in-time availability
and never marks data as backtest-ready.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd


_REQUIRED = {
    "symbol", "trade_date", "open", "high", "low", "close",
    "volume", "amount", "source", "data_version", "fetched_at",
}


def _default_loader():
    try:
        from cnequity.query import load
    except ImportError as exc:
        raise RuntimeError(
            "CNEquity is not installed; install cnequity to use this provider"
        ) from exc
    return load


def normalize_daily_bars(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate CNEquity raw daily bars without fabricating PIT metadata."""
    # CNEquity query.load() returns Polars; convert without an Arrow bridge.
    if not isinstance(frame, pd.DataFrame):
        if hasattr(frame, "to_dicts"):
            frame = pd.DataFrame(frame.to_dicts())
        else:
            raise TypeError("CNEquity daily_bars must be a DataFrame-like table")
    missing = sorted(_REQUIRED - set(frame.columns))
    if missing:
        raise ValueError("CNEquity daily_bars missing columns: " + ",".join(missing))
    if frame.empty:
        raise ValueError("CNEquity daily_bars returned no rows")

    out = frame.copy()
    out["symbol"] = out["symbol"].astype(str).str.strip()
    if out["symbol"].eq("").any():
        raise ValueError("CNEquity daily_bars contains blank symbols")
    out["trade_date"] = pd.to_datetime(out["trade_date"], errors="coerce")
    if out["trade_date"].isna().any():
        raise ValueError("CNEquity daily_bars contains invalid trade_date")
    if out.duplicated(["symbol", "trade_date"]).any():
        raise ValueError("CNEquity daily_bars contains duplicate symbol/trade_date")

    numeric_cols = ["open", "high", "low", "close", "volume", "amount"]
    numeric = out[numeric_cols].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any() or not np.isfinite(numeric.to_numpy()).all():
        raise ValueError("CNEquity daily_bars contains missing/non-finite OHLCVA")
    if (numeric[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError("CNEquity daily_bars contains non-positive prices")
    if (numeric[["volume", "amount"]] < 0).any().any():
        raise ValueError("CNEquity daily_bars contains negative volume/amount")
    if (numeric["high"] < numeric[["open", "low", "close"]].max(axis=1)).any():
        raise ValueError("CNEquity daily_bars has invalid high")
    if (numeric["low"] > numeric[["open", "high", "close"]].min(axis=1)).any():
        raise ValueError("CNEquity daily_bars has invalid low")

    versions = out["data_version"].astype(str)
    if not versions.eq("v2").all():
        raise ValueError(
            "CNEquity daily_bars requires data_version=v2 to enforce volume in shares"
        )
    if out["source"].isna().any() or out["source"].astype(str).str.strip().eq("").any():
        raise ValueError("CNEquity daily_bars missing source lineage")
    fetched = pd.to_datetime(out["fetched_at"], errors="coerce", utc=True)
    if fetched.isna().any():
        raise ValueError("CNEquity daily_bars has invalid fetched_at")

    for column in numeric_cols:
        out[column] = numeric[column]
    out["fetched_at"] = fetched
    # fetched_at records retrieval, not when the observation first became
    # public. Never pass these rows to the strict historical backtest as PIT.
    out["available_time"] = pd.NaT
    out["historical_pit_eligible"] = False
    out["pit_exclusion_reason"] = "retrieval_time_is_not_historical_availability"
    return out.sort_values(["symbol", "trade_date"]).reset_index(drop=True)


def load_daily_bars(
    symbols: Sequence[str],
    *,
    start: str | None = None,
    end: str | None = None,
    loader: Callable | None = None,
) -> pd.DataFrame:
    """Load bounded CNEquity symbols; symbols use e.g. 600000.SH."""
    normalized_symbols = sorted({str(symbol).strip() for symbol in symbols if str(symbol).strip()})
    if not normalized_symbols:
        raise ValueError("at least one CNEquity symbol is required")
    load = loader or _default_loader()
    kwargs = {"symbols": normalized_symbols, "adjust": None}
    if start is not None:
        kwargs["start"] = start
    if end is not None:
        kwargs["end"] = end
    frame = load("daily_bars", **kwargs)
    return normalize_daily_bars(frame)
