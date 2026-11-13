"""Fail-closed assembler for forward-archived M3 market observations.

This module builds a *current-decision* history from prospective daily breadth
and limit-pool archives plus index daily bars. It does not make historical
point-in-time backtests eligible.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.model.market_state import (
    apply_state_cooldown, calculate_state_scores, market_environment_score,
)


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"corrupt archive {path}:{line_no}") from exc
    return rows


def _utc(value, label: str) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        raise ValueError(f"{label} must be timezone-aware")
    return ts.tz_convert("UTC")


def assemble_m3_history(
    archive_root: str | Path,
    *,
    index_symbol: str = "000001",
    decision_time,
    min_breadth_coverage: float = 0.95,
) -> dict:
    """Return READY only when 60 consecutive fully observed market sessions exist."""
    cutoff = _utc(decision_time, "decision_time")
    if not 0 < min_breadth_coverage <= 1:
        raise ValueError("min_breadth_coverage must be in (0, 1]")
    root = Path(archive_root)

    index_rows = _read_jsonl(root / "index_daily" / f"{str(index_symbol).zfill(6)}.jsonl")
    if not index_rows:
        return {"status": "INSUFFICIENT_DATA", "reason": "index_archive_missing", "history": pd.DataFrame()}
    idx = pd.DataFrame(index_rows)
    required_index = {"trade_date", "close", "available_time"}
    if required_index - set(idx.columns):
        return {"status": "INSUFFICIENT_DATA", "reason": "index_schema_incomplete", "history": pd.DataFrame()}
    idx["trade_date"] = pd.to_datetime(idx["trade_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    idx["available_time"] = pd.to_datetime(idx["available_time"], errors="coerce", utc=True)
    idx["close"] = pd.to_numeric(idx["close"], errors="coerce")
    idx = idx.loc[(idx.available_time <= cutoff) & idx.trade_date.notna()].copy()
    if idx.trade_date.duplicated().any():
        return {"status": "INVALID_DATA", "reason": "duplicate_index_trade_date", "history": pd.DataFrame()}
    idx = idx.sort_values("trade_date")
    if idx.close.isna().any() or (idx.close <= 0).any():
        return {"status": "INVALID_DATA", "reason": "invalid_index_close", "history": pd.DataFrame()}
    idx["ma20"] = idx.close.rolling(20, min_periods=20).mean()
    idx["ma60"] = idx.close.rolling(60, min_periods=60).mean()
    idx["vol20"] = idx.close.pct_change().rolling(20, min_periods=20).std(ddof=0)

    breadth_by_date = {}
    for path in sorted((root / "market_breadth").glob("*.jsonl")):
        day = path.stem
        snapshots = _read_jsonl(path)
        eligible = []
        for row in snapshots:
            captured = _utc(row.get("snapshot_time"), "breadth snapshot_time")
            available_value = row.get("available_time", row.get("snapshot_time"))
            available = _utc(available_value, "breadth available_time")
            local = captured.tz_convert("Asia/Shanghai")
            if (
                local.strftime("%Y-%m-%d") == day
                and local.strftime("%H:%M") >= "15:30"
                and captured <= cutoff
                and available <= cutoff
            ):
                eligible.append((captured, row))
        if eligible:
            preferred = [
                item for item in eligible
                if item[1].get("source_id") == "kaipanhong.market_emotion_kph"
            ]
            breadth_by_date[day] = max(
                preferred or eligible, key=lambda item: item[0]
            )[1]

    sentiment_by_date = {}
    for path in sorted((root / "market_sentiment").glob("*.jsonl")):
        for row in _read_jsonl(path):
            day = str(row.get("trade_date", ""))
            captured = _utc(row.get("capture_time"), "sentiment capture_time")
            available_value = row.get("available_time", row.get("capture_time"))
            available = _utc(available_value, "sentiment available_time")
            local = captured.tz_convert("Asia/Shanghai")
            if (
                local.strftime("%Y-%m-%d") == day
                and local.strftime("%H:%M") >= "15:30"
                and captured <= cutoff
                and available <= cutoff
            ):
                old = sentiment_by_date.get(day)
                is_preferred = row.get("source_id") == "kaipanhong.market_emotion_kph"
                old_preferred = (
                    old is not None
                    and old[1].get("source_id") == "kaipanhong.market_emotion_kph"
                )
                if (
                    old is None
                    or (is_preferred and not old_preferred)
                    or (is_preferred == old_preferred and captured > old[0])
                ):
                    sentiment_by_date[day] = (captured, row)

    merged = idx.copy()
    merged["breadth"] = np.nan
    merged["amount"] = np.nan
    merged["limit_up_cnt"] = np.nan
    merged["limit_down_cnt"] = np.nan
    merged["snapshot_available_time"] = pd.Series(
        pd.NaT, index=merged.index, dtype="datetime64[ns, UTC]"
    )
    reasons = []
    for i, row in merged.iterrows():
        day = row.trade_date
        breadth = breadth_by_date.get(day)
        sentiment_pair = sentiment_by_date.get(day)
        if breadth is None or sentiment_pair is None:
            continue
        sentiment_time, sentiment = sentiment_pair
        breadth_time = _utc(breadth.get("snapshot_time"), "breadth snapshot_time")
        coverage = pd.to_numeric(breadth.get("breadth_coverage"), errors="coerce")
        amount = pd.to_numeric(breadth.get("total_amount"), errors="coerce")
        ratio = pd.to_numeric(breadth.get("breadth"), errors="coerce")
        if pd.isna(coverage) or coverage < min_breadth_coverage:
            reasons.append(f"{day}:breadth_coverage_below_minimum")
            continue
        if breadth.get("total_amount_status") != "complete_provider_snapshot_sum" or pd.isna(amount) or amount < 0:
            reasons.append(f"{day}:turnover_incomplete")
            continue
        if pd.isna(ratio) or not 0 <= ratio <= 1:
            reasons.append(f"{day}:breadth_invalid")
            continue
        up = pd.to_numeric(sentiment.get("limit_up_count"), errors="coerce")
        down = pd.to_numeric(sentiment.get("limit_down_count"), errors="coerce")
        if pd.isna(up) or pd.isna(down) or up < 0 or down < 0:
            reasons.append(f"{day}:limit_pool_counts_missing")
            continue
        merged.at[i, "breadth"] = float(ratio)
        merged.at[i, "amount"] = float(amount)
        merged.at[i, "limit_up_cnt"] = float(up)
        merged.at[i, "limit_down_cnt"] = float(down)
        merged.at[i, "snapshot_available_time"] = max(breadth_time, sentiment_time)

    merged["amount_ma20"] = merged.amount.rolling(20, min_periods=20).mean()
    merged["available_time"] = merged[["available_time", "snapshot_available_time"]].max(axis=1)
    complete = merged.dropna(subset=[
        "ma20", "ma60", "vol20", "breadth", "amount", "amount_ma20",
        "limit_up_cnt", "limit_down_cnt", "available_time",
    ]).copy()
    complete = complete.sort_values("trade_date").reset_index(drop=True)

    # Sixty percentile observations each need an MA60/volatility feature row.
    # Without index warm-up predating the archive, that means 119 consecutive
    # aligned daily market snapshots. Never skip a missing session in the window.
    last_index_days = idx.trade_date.tail(119).tolist()
    snapshot_complete = merged.dropna(subset=[
        "breadth", "amount", "limit_up_cnt", "limit_down_cnt",
        "snapshot_available_time",
    ])
    snapshot_complete_days = set(snapshot_complete.trade_date.tolist())
    missing_days = [day for day in last_index_days if day not in snapshot_complete_days]
    if len(idx) < 119 or missing_days:
        return {
            "status": "INSUFFICIENT_DATA",
            "reason": "fewer_than_119_complete_consecutive_sessions",
            "valid_sessions": int(len(complete)),
            "required_sessions": 119,
            "missing_recent_dates": missing_days[-10:],
            "quality_warnings": reasons[-10:],
            "history": complete,
            "historical_backtest_eligible": False,
        }
    # The 60-day percentile window runs on rows that already have MA60 and
    # volatility features. With no pre-existing index warm-up, this requires
    # about 119 aligned sessions, not merely 60 raw daily snapshots.
    if len(complete) < 60:
        return {
            "status": "INSUFFICIENT_DATA",
            "reason": "fewer_than_60_scored_feature_observations",
            "valid_sessions": int(len(complete)),
            "required_sessions": 60,
            "missing_recent_dates": [],
            "quality_warnings": reasons[-10:],
            "history": complete,
            "historical_backtest_eligible": False,
        }

    scored = calculate_state_scores(complete, decision_time=cutoff)
    scored = apply_state_cooldown(scored)
    latest = scored.iloc[-1]
    state = latest.market_state
    ready = state in {"RISK_ON", "NEUTRAL", "RISK_OFF"}
    return {
        "status": "READY" if ready else "STATE_CONFIRMATION_PENDING",
        "reason": None if ready else "two_consecutive_sessions_not_yet_confirmed",
        "valid_sessions": int(len(complete)),
        "latest_trade_date": str(latest.trade_date.date()),
        "market_state": state,
        "state_score": None if pd.isna(latest.StateScore) else float(latest.StateScore),
        "market_environment_score": (
            market_environment_score(latest)
            if state in {"RISK_ON", "NEUTRAL", "RISK_OFF"} else None
        ),
        "position_scale": None if pd.isna(latest.position_scale) else float(latest.position_scale),
        "history": scored,
        "historical_backtest_eligible": False,
        "quality_warnings": reasons[-10:],
    }
