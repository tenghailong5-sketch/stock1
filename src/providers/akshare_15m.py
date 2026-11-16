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
    "时间": "bar_time", "日期": "bar_time", "day": "bar_time", "开盘": "open",
    "最高": "high", "最低": "low", "收盘": "close",
    "成交量": "volume", "成交额": "amount",
    "bar_time": "bar_time", "open": "open", "high": "high",
    "low": "low", "close": "close", "volume": "volume", "amount": "amount",
}
BAR_REQUIRED = {"bar_time", "open", "high", "low", "close", "volume"}
REQUIRED = BAR_REQUIRED | {"amount"}


def normalize_15m_bars(raw: pd.DataFrame, *, symbol: str, captured_at=None, source_id="akshare.stock_zh_a_hist_min_em") -> pd.DataFrame:
    if raw is None or not isinstance(raw, pd.DataFrame) or raw.empty:
        raise ValueError("15-minute provider returned no bars")
    frame = raw.rename(columns={key: value for key, value in ALIASES.items() if key in raw.columns}).copy()
    missing = BAR_REQUIRED - set(frame.columns)
    if missing:
        raise ValueError("15-minute bars missing: " + ",".join(sorted(missing)))
    # Sina's minute endpoint does not expose traded value; keep it missing.
    # Never manufacture turnover from close * volume.
    if "amount" not in frame.columns:
        frame["amount"] = np.nan
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
    if frame[["open", "high", "low", "close", "volume"]].isna().any().any():
        raise ValueError("15-minute OHLCV contains missing/non-numeric core values")
    if (~np.isfinite(frame[["open", "high", "low", "close", "volume"]])).any().any():
        raise ValueError("15-minute OHLCV contains non-finite core values")
    observed_amount = frame["amount"].dropna()
    if not observed_amount.empty and (~np.isfinite(observed_amount)).any():
        raise ValueError("15-minute traded value contains non-finite values")
    if (frame[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError("15-minute prices must be positive")
    if (frame["volume"] < 0).any() or (frame["amount"].dropna() < 0).any():
        raise ValueError("15-minute volume/observed amount cannot be negative")
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
    frame["source_id"] = str(source_id)
    frame["turnover_observed"] = frame["amount"].notna()
    frame["availability_basis"] = "retrieval_time_only"
    frame["historical_pit_eligible"] = False
    return frame.reset_index(drop=True)


def confirm_15m_breakout(
    bars: pd.DataFrame, *, lookback_bars: int = 4,
    minimum_volume_multiple: float = 1.0,
    max_bar_age_minutes: float = 30.0,
) -> dict:
    """Confirm a completed 15m breakout; thresholds are explicit research config."""
    if isinstance(lookback_bars, bool) or not isinstance(lookback_bars, int) or lookback_bars < 2:
        raise ValueError("lookback_bars must be an integer >= 2")
    if (not np.isfinite(minimum_volume_multiple) or minimum_volume_multiple <= 0):
        raise ValueError("minimum_volume_multiple must be finite and positive")
    if (not np.isfinite(max_bar_age_minutes) or max_bar_age_minutes <= 0):
        raise ValueError("max_bar_age_minutes must be finite and positive")
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
    capture_local = capture.tz_convert("Asia/Shanghai")
    bar_end_local = bar_end.tz_convert("Asia/Shanghai")
    seconds_since_midnight = (
        capture_local.hour * 3600
        + capture_local.minute * 60
        + capture_local.second
    )
    in_entry_session = (
        9 * 3600 + 45 * 60 <= seconds_since_midnight <= 11 * 3600 + 30 * 60
        or 13 * 3600 + 15 * 60 <= seconds_since_midnight <= 15 * 3600
    )
    if not in_entry_session:
        return {
            "status": "OUTSIDE_15M_ENTRY_WINDOW", "entry_ready": False,
            "reason": "15m_entry_confirmation_requires_active_trading_session",
            "completed_bars": int(len(x)),
            "available_time": capture_local.isoformat(),
            "historical_pit_eligible": False,
        }
    bar_age_minutes = (
        capture_local - bar_end_local
    ).total_seconds() / 60.0
    if (
        bar_end_local.date() != capture_local.date()
        or bar_age_minutes < 0
        or bar_age_minutes > float(max_bar_age_minutes)
    ):
        return {
            "status": "STALE_15M_CONFIRMATION", "entry_ready": False,
            "reason": "latest_15m_bar_not_fresh_for_current_session",
            "completed_bars": int(len(x)),
            "bar_end_time": bar_end_local.isoformat(),
            "available_time": capture_local.isoformat(),
            "bar_age_minutes": bar_age_minutes,
            "max_bar_age_minutes": float(max_bar_age_minutes),
            "historical_pit_eligible": False,
        }
    prior_high = float(previous.high.max())
    prior_volume = float(previous.volume.mean())
    volume_multiple = float(latest.volume / prior_volume) if prior_volume > 0 else float("nan")
    checks = {
        "close_above_prior_high": bool(float(latest.close) > prior_high),
        "volume_expansion": bool(np.isfinite(volume_multiple) and volume_multiple >= minimum_volume_multiple),
        "positive_candle": bool(float(latest.close) >= float(latest.open)),
        "positive_turnover": bool(float(latest.amount) > 0 if pd.notna(latest.amount) else float(latest.volume) > 0),
    }
    passed = all(checks.values())
    entry_reference_price = float(latest.close)
    breakout_trigger_price = prior_high
    stop_reference_price = min(float(latest.low), prior_high)
    stop_distance_pct = (
        (entry_reference_price - stop_reference_price) / entry_reference_price * 100.0
        if entry_reference_price > stop_reference_price else None
    )
    invalidation_condition = (
        f"15分钟收盘跌破结构止损参考位{stop_reference_price:.4f}，"
        f"或重新跌回突破触发位{breakout_trigger_price:.4f}下方"
    )
    return {
        "status": "CONFIRMED_15M_BREAKOUT" if passed else "NO_15M_CONFIRMATION",
        "entry_ready": passed,
        "reason": "all_15m_checks_passed" if passed else ";".join(k for k, v in checks.items() if not v),
        "completed_bars": int(len(x)),
        "bar_time": pd.Timestamp(latest.bar_time).isoformat(),
        "bar_end_time": bar_end.isoformat(),
        "available_time": capture.isoformat(),
        "latest_close": float(latest.close),
        "entry_reference_price": entry_reference_price,
        "breakout_trigger_price": breakout_trigger_price,
        "stop_reference_price": stop_reference_price,
        "stop_distance_pct": stop_distance_pct,
        "invalidation_condition": invalidation_condition,
        "prior_lookback_high": prior_high,
        "volume_multiple": volume_multiple if np.isfinite(volume_multiple) else None,
        "checks": checks,
        "source_id": str(latest.get("source_id", "unknown")),
        "turnover_basis": "observed_amount" if pd.notna(latest.amount) else "positive_volume_only_amount_unavailable",
        "availability_basis": str(latest.get("availability_basis", "unknown")),
        "historical_pit_eligible": False,
    }


def fetch_live_15m_confirmation(
    symbol: str, *, captured_at=None, fetcher=None, fallback_fetcher=None
) -> dict:
    """Fetch 15m bars with Eastmoney first and Sina fallback on provider failure."""
    import akshare as ak

    capture = pd.Timestamp.now(tz="UTC") if captured_at is None else pd.Timestamp(captured_at)
    if capture.tzinfo is None:
        raise ValueError("captured_at must be timezone-aware")
    local_now = capture.tz_convert("Asia/Shanghai")
    start = (local_now - pd.Timedelta(days=7)).strftime("%Y-%m-%d %H:%M:%S")
    end = local_now.strftime("%Y-%m-%d %H:%M:%S")
    primary = fetcher or ak.stock_zh_a_hist_min_em
    try:
        raw = primary(
            symbol=str(symbol).zfill(6), period="15",
            start_date=start, end_date=end, adjust="",
        )
        source_id = "akshare.stock_zh_a_hist_min_em"
    except Exception as primary_exc:
        secondary = fallback_fetcher or ak.stock_zh_a_minute
        code = str(symbol).strip().zfill(6)
        if code.startswith("6"):
            sina_symbol = "sh" + code
        elif code.startswith(("0", "2", "3")):
            sina_symbol = "sz" + code
        else:
            raise RuntimeError(
                f"Eastmoney 15m failed and Sina fallback does not support symbol {code}: "
                f"{type(primary_exc).__name__}:{str(primary_exc)[:180]}"
            ) from primary_exc
        try:
            raw = secondary(symbol=sina_symbol, period="15", adjust="")
            source_id = "akshare.stock_zh_a_minute_sina"
        except Exception as fallback_exc:
            raise RuntimeError(
                "15m providers failed; Eastmoney="
                f"{type(primary_exc).__name__}:{str(primary_exc)[:120]}; Sina="
                f"{type(fallback_exc).__name__}:{str(fallback_exc)[:120]}"
            ) from fallback_exc
    normalized = normalize_15m_bars(
        raw, symbol=symbol, captured_at=capture, source_id=source_id
    )
    return confirm_15m_breakout(normalized)
