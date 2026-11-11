"""Forward-only snapshots of Eastmoney limit-up/limit-down pools.

These are provider pool snapshots, not tick-level exchange event feeds.
The date argument is YYYYMMDD and provider coverage is limited; no historical
backfill or inferred limit-price classification is performed here.
"""
from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone

import pandas as pd


POOL_APIS = {
    "limit_up": "stock_zt_pool_em",
    "limit_down": "stock_zt_pool_dtgc_em",
    "limit_up_opened": "stock_zt_pool_zbgc_em",
    "previous_limit_up": "stock_zt_pool_previous_em",
    "strong": "stock_zt_pool_strong_em",
}


def _captured(value=None) -> pd.Timestamp:
    ts = pd.Timestamp.now(tz="UTC") if value is None else pd.Timestamp(value)
    if ts.tzinfo is None:
        raise ValueError("capture timestamps must be timezone-aware")
    return ts.tz_convert("UTC")


def normalize_limit_pool_snapshot(
    frame: pd.DataFrame,
    *,
    event_type: str,
    trade_date: str,
    captured_at=None,
) -> pd.DataFrame:
    """Normalize one date-scoped provider pool; empty-with-schema means zero rows."""
    if event_type not in POOL_APIS:
        raise ValueError(f"unsupported limit pool event_type: {event_type}")
    if not isinstance(trade_date, str) or not re.fullmatch(r"\d{8}", trade_date):
        raise ValueError("trade_date must be YYYYMMDD")
    if frame is None:
        raise ValueError(f"{event_type} provider returned None")
    if "代码" not in frame.columns:
        raise ValueError(f"{event_type} provider schema missing 代码")
    raw = frame.copy()
    if not raw.empty:
        symbols = raw["代码"].astype(str).str.zfill(6)
        if symbols.str.fullmatch(r"\d{6}").eq(False).any():
            raise ValueError(f"{event_type} contains invalid stock codes")
        if symbols.duplicated().any():
            raise ValueError(f"{event_type} contains duplicate stock codes")
        raw["代码"] = symbols
    records = json.loads(raw.to_json(orient="records", force_ascii=False, date_format="iso"))
    captured = _captured(captured_at)
    payload = {
        "event_type": event_type,
        "trade_date": f"{trade_date[:4]}-{trade_date[4:6]}-{trade_date[6:]}",
        "capture_time": captured.isoformat(),
        "source_id": f"akshare.{POOL_APIS[event_type]}",
        "snapshot_scope": "date_scoped_provider_pool_snapshot",
        "historical_pit_eligible": False,
        "provider_row_count": len(records),
        "records": records,
    }
    canonical = json.dumps(
        {key: value for key, value in payload.items() if key != "capture_time"},
        ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    )
    payload["source_revision"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return pd.DataFrame([payload])


def fetch_limit_pool_snapshots(ak_module, *, trade_date: str, captured_at=None) -> pd.DataFrame:
    """Fetch all configured date-scoped pools; any API/schema error aborts the batch."""
    frames = []
    for event_type, api_name in POOL_APIS.items():
        api = getattr(ak_module, api_name, None)
        if not callable(api):
            raise ValueError(f"AKShare API unavailable: {api_name}")
        raw = api(date=trade_date)
        frames.append(normalize_limit_pool_snapshot(
            raw, event_type=event_type, trade_date=trade_date, captured_at=captured_at
        ))
    return pd.concat(frames, ignore_index=True)
