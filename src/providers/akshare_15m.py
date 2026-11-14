"""15-minute entry confirmation for live A-share research scans.

Bars are stamped with their actual retrieval time. They may confirm a current
live setup after a bar has closed, but they are not historical PIT evidence.
The module emits confirmation diagnostics only; it never creates an M5 score.
"""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd


ALIASES = {
    "时间": "bar_time", "日期": "bar_time", "开盘": "open",
    "最高": "high", "最低": "low", "收盘": "close",
    "成交量": "volume", "成交额": "amount",
    "bar_time": "bar_time", "open": "open", "high": "high",
    "low": "low", "close": "close", "volume": "volume", "amount": "amount",
}
REQUIRED = {"bar_time", "open", "high", "low", "close", "volume", "amount"}


def normalize_15m_bars(raw: pd.DataFrame, *, symbol: str, captured_at=None) -> pd.DataFrame:
    if raw is None or not isinstance(raw, pd.DataFrame) or raw.empty:
        raise ValueError("15-minute provider returned no bars")
    frame = raw.rename(columns={key: value for key, value in ALIASES.items() if key in raw.columns}).copy()
    missing = REQUIRED - set(frame.columns)
    if missing:
        raise ValueError("15-minute bars missing: " + ",".join(sorted(missing)))
    stamp = pd.Timestamp.now(tz="UTC") if captured_at is None else pd.Timestamp(captured_at)
    if stamp.tzinfo is None:
        raise ValueError("captured_at must be timezone-aware")
    stamp = stamp.tz_convert("UTC")
    times = pd.to_datetime(frame["bar_time"], errors="coerce")
    if times.isna().any():
        raise ValueError("15-minute bar_time contains invalid timestamps")
    if times.dt.tz is None:
        times = times.dt.tz_localize("Asia/Shanghai")
    else:
        times = times.dt.tz_convert("Asia/Shanghai")
    frame["bar_time"] = times
    for column in ("open", "high", "low", "close", "volume", "amount"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if frame[list(("open", "high", "low", "close", "volume", "amount"))].isna().any().any():
        raise ValueError("15-minute OHLCV contains missing/non-numeric values")
    if (~np.isfinite(frame[["open", "high", "low", "close", "volume", "amount"]])).any().any():
        raise ValueError("15-minute OHLCV contains non-finite values")
    if (frame[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError("15-minute prices must be positive")
    if (frame[["volume", "amount"]] < 0).any().any():
        raise ValueError("15-minute volume/amount cannot be negative")
    if (frame["high"] < frame[["open", "low", "close"]].max(axis=1)).any():
        raise ValueError("15-minute high is inconsistent")
    if (frame["low"] > frame[["open", "high", "close"]].min(axis=1)).any():
        raise ValueError("15-minute low is inconsistent")
    if frame["bar_time"].duplicated().any():
        raise ValueError("duplicate 15-minute bar timestamps")
    frame = frame.sort_values("bar_time").reset_index(drop=True)
    # AKShare's minute timestamps identify the bar start; only fully closed bars survive.
    frame["bar_end_time"] = frame["bar_time"] + pd.Timedelta(minutes=15)
    frame = frame.loc[frame["bar_end_time"].dt.tz_convert("UTC") <= stamp].copy()
    frame["symbol"] = str(symbol).strip().zfill(6)
    frame["available_time"] = stamp
    frame["source_id"] = "akshare.stock_zh_a_hist_min_em"
    frame["availability_basis"] = "retrieval_time_only"
    frame["historical_pit_eligible"] = False
    return frame.reset_index(drop=True)


def confirm_15m_breakout(
    bars: pd.DataFrame, *, lookback_bars: int = 4,
    minimum_volume_multiple: float = 1.0,
) -> dict:
    """Confirm a completed 15m breakout; thresholds are explicit research config."""
    if isinstance(lookback_bars, bool) or not isinstance(lookback_bars, int) or lookback_bars < 2:
        raise ValueError("lookback_bars must be an integer >= 2")
    if (not np.isfinite(minimum_volume_multiple) or minimum_volume_multiple <= 0):
        raise ValueError("minimum_volume_multiple must be finite and positive")
    required = REQUIRED | {"bar_end_time", "available_time", "symbol"}
    missing = required - set(bars.columns)
    if missing:
        raise ValueError("normalized 15-minute bars missing: " + ",".join(sorted(missing)))
    if len(bars) < lookback_bars + 1:
        return {
            "status": "INSUFFICIENT_15M_HISTORY", "entry_ready": False,
            "reason": "not_enough_completed_15m_bars", "completed_bars": int(len(bars)),
            "historical_pit_eligible": False,
        }
    x = bars.sort_values("bar_time").reset_index(drop=True)
    latest = x.iloc[-1]
    previous = x.iloc[-lookback_bars-1:-1]
    capture = pd.Timestamp(latest.available_time)
    bar_end = pd.Timestamp(latest.bar_end_time)
    if capture.tzinfo is None or bar_end.tzinfo is None:
        raise ValueError("15-minute bar and availability timestamps must be timezone-aware")
    if bar_end.tz_convert("UTC") > capture.tz_convert("UTC"):
        return {
            "status": "PARTIAL_BAR_REJECTED", "entry_ready": False,
            "reason": "latest_15m_bar_not_closed", "completed_bars": int(len(x)),
            "historical_pit_eligible": False,
        }
    prior_high = float(previous.high.max())
    prior_volume = float(previous.volume.mean())
    volume_multiple = float(latest.volume / prior_volume) if prior_volume > 0 else float("nan")
    checks = {
        "close_above_prior_high": bool(float(latest.close) > prior_high),
        "volume_expansion": bool(np.isfinite(volume_multiple) and volume_multiple >= minimum_volume_multiple),
        "positive_candle": bool(float(latest.close) >= float(latest.open)),
        "positive_turnover": bool(float(latest.amount) > 0),
    }
    passed = all(checks.values())
    return {
        "status": "CONFIRMED_15M_BREAKOUT" if passed else "NO_15M_CONFIRMATION",
        "entry_ready": passed,
        "reason": "all_15m_checks_passed" if passed else ";".join(k for k, v in checks.items() if not v),
        "completed_bars": int(len(x)),
        "bar_time": pd.Timestamp(latest.bar_time).isoformat(),
        "bar_end_time": bar_end.isoformat(),
        "available_time": capture.isoformat(),
        "latest_close": float(latest.close),
        "prior_lookback_high": prior_high,
        "volume_multiple": volume_multiple if np.isfinite(volume_multiple) else None,
        "checks": checks,
        "source_id": str(latest.get("source_id", "unknown")),
        "availability_basis": str(latest.get("availability_basis", "unknown")),
        "historical_pit_eligible": False,
    }


def fetch_live_15m_confirmation(symbol: str, *, captured_at=None, fetcher=None) -> dict:
    """Fetch recent 15m bars from AKShare and return a live-only confirmation."""
    if fetcher is None:
        import akshare as ak
        fetcher = ak.stock_zh_a_hist_min_em
    capture = pd.Timestamp.now(tz="UTC") if captured_at is None else pd.Timestamp(captured_at)
    if capture.tzinfo is None:
        raise ValueError("captured_at must be timezone-aware")
    local_now = capture.tz_convert("Asia/Shanghai")
    start = (local_now - pd.Timedelta(days=7)).strftime("%Y-%m-%d %H:%M:%S")
    end = local_now.strftime("%Y-%m-%d %H:%M:%S")
    raw = fetcher(
        symbol=str(symbol).zfill(6), period="15",
        start_date=start, end_date=end, adjust="",
    )
    normalized = normalize_15m_bars(raw, symbol=symbol, captured_at=capture)
    return confirm_15m_breakout(normalized)
