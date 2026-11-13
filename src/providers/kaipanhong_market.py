"""Historical market-wide breadth and limit-event snapshots from KaipanHong.

The provider returns aggregate daily observations for historical dates. Values
are usable for current-decision context after retrieval, but are never admitted
as historical point-in-time backtest evidence.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone

import pandas as pd
import requests


URL = "https://apphis.kaipanhong.com/w1/api/index.php"
HEADERS = {
    "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
    "User-Agent": "Dalvik/2.1.0 (Linux; U; Android 12; Mobile)",
    "Accept-Encoding": "gzip",
}
BASE_PARAMS = {
    "PhoneOSNew": "1",
    "DeviceID": "1a609dd6-b2b8-3bf9-ac40-a77581551454",
    "VerSion": "6.0.6",
    "Token": "0",
    "UserID": "0",
    "Red": "1",
    "apiv": "w45",
}


def _integer(info: dict, key: str) -> int:
    value = pd.to_numeric(info.get(key), errors="coerce")
    if pd.isna(value) or float(value) < 0 or not float(value).is_integer():
        raise ValueError(f"invalid historical market field: {key}")
    return int(value)


def _amount(info: dict) -> float:
    value = pd.to_numeric(info.get("qscln"), errors="coerce")
    if pd.isna(value) or not float(value) > 0:
        raise ValueError("invalid historical total market turnover")
    return float(value)


def fetch_historical_market_context(
    trade_date: str,
    *,
    expected_market_count: int,
    fetched_at=None,
    post=requests.post,
    min_coverage: float = 0.95,
) -> tuple[dict, dict]:
    """Fetch one historical market breadth + limit-count observation.

    snapshot_time/capture_time describe the historical observation date.
    available_time records when this system actually retrieved the data.
    Historical backtest eligibility remains explicitly false.
    """
    day = pd.Timestamp(trade_date)
    if pd.isna(day) or day.strftime("%Y-%m-%d") != str(trade_date):
        raise ValueError("trade_date must use YYYY-MM-DD")
    if expected_market_count < 1:
        raise ValueError("expected_market_count must be positive")
    if not 0 < min_coverage <= 1:
        raise ValueError("min_coverage must be in (0, 1]")

    retrieved = pd.Timestamp(
        fetched_at if fetched_at is not None else datetime.now(timezone.utc)
    )
    if retrieved.tzinfo is None:
        raise ValueError("fetched_at must be timezone-aware")
    retrieved = retrieved.tz_convert("UTC")
    local_retrieved = retrieved.tz_convert("Asia/Shanghai")
    if day.date() > local_retrieved.date():
        raise ValueError("cannot fetch a future market date")

    params = {
        **BASE_PARAMS,
        "a": "HisZhangFuDetail",
        "c": "HisHomeDingPan",
        "Day": str(trade_date),
    }
    response = post(URL, headers=HEADERS, data=params, timeout=10)
    response.raise_for_status()
    payload = response.json()
    if str(payload.get("errcode")) != "0":
        raise RuntimeError(
            f"KaipanHong historical endpoint errcode={payload.get('errcode')}"
        )
    info = payload.get("info")
    if not isinstance(info, dict) or not info:
        raise ValueError("KaipanHong historical endpoint returned empty info")

    advances = _integer(info, "SZJS")
    declines = _integer(info, "XDJS")
    unchanged = _integer(info, "0")
    limit_up = _integer(info, "ZT")
    limit_down = _integer(info, "DT")
    valid_count = advances + declines + unchanged
    coverage = valid_count / expected_market_count
    if valid_count <= 0 or coverage < min_coverage:
        raise ValueError(
            f"historical market breadth coverage below minimum: "
            f"{valid_count}/{expected_market_count}={coverage:.1%}"
        )
    if coverage > 1.05:
        raise ValueError(
            f"historical market breadth count implausibly exceeds universe: {coverage:.1%}"
        )
    total_amount = _amount(info)

    observation_time = pd.Timestamp(
        f"{trade_date} 15:40:00", tz="Asia/Shanghai"
    ).tz_convert("UTC")
    if day.date() == local_retrieved.date():
        observation_time = retrieved

    source_id = "kaipanhong.market_emotion_kph"
    values = {
        "trade_date": str(trade_date),
        "advances": advances,
        "declines": declines,
        "unchanged": unchanged,
        "limit_up_count": limit_up,
        "limit_down_count": limit_down,
        "total_amount": total_amount,
        "valid_quote_count": valid_count,
        "expected_market_count": expected_market_count,
    }
    revision = hashlib.sha256(
        json.dumps(values, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()

    breadth = {
        "trade_date": str(trade_date),
        "snapshot_time": observation_time.isoformat(),
        "available_time": retrieved.isoformat(),
        "provider_row_count": valid_count,
        "valid_quote_count": valid_count,
        "missing_quote_count": max(0, expected_market_count - valid_count),
        "advances": advances,
        "declines": declines,
        "unchanged": unchanged,
        "breadth_coverage": min(1.0, coverage),
        "breadth": advances / valid_count,
        "total_amount": total_amount,
        "total_amount_status": "complete_provider_market_aggregate",
        "snapshot_scope": "historical_market_aggregate",
        "availability_basis": "retrieval_time_only",
        "historical_pit_eligible": False,
        "source_id": source_id,
        "source_revision": revision,
    }
    sentiment = {
        "trade_date": str(trade_date),
        "capture_time": observation_time.isoformat(),
        "available_time": retrieved.isoformat(),
        "limit_up_count": limit_up,
        "limit_down_count": limit_down,
        "snapshot_scope": "historical_market_aggregate",
        "availability_basis": "retrieval_time_only",
        "historical_pit_eligible": False,
        "source_id": source_id,
        "source_revision": revision,
    }
    return breadth, sentiment
