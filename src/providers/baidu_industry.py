"""Baidu Gushitong industry membership adapter (free public endpoint).

This is an independent per-security source. It retains Baidu's original
industry code/name and retrieval-time availability; it does not pretend that
Baidu labels are identical to Eastmoney or SW classifications.
"""
from __future__ import annotations

import hashlib
import json
import time
import uuid
from datetime import datetime, timezone
from urllib.parse import parse_qs

import pandas as pd
import requests


URL = "https://finance.pae.baidu.com/api/getrelatedblock"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120 Safari/537.36",
    "Accept": "application/vnd.finance-web.v1+json, application/json",
    "Referer": "https://gushitong.baidu.com/",
    "Origin": "https://gushitong.baidu.com",
}


def fetch_baidu_industry_membership_snapshot(
    symbols, *, session=None, clock=None, batch_size: int = 50,
    timeout: int = 25, pause_seconds: float = 0.25, sleep_fn=time.sleep,
) -> pd.DataFrame:
    """Fetch Baidu industry labels for the supplied universe in bounded batches."""
    if batch_size < 1 or batch_size > 100:
        raise ValueError("batch_size must be between 1 and 100")
    requested = sorted({str(x).strip().zfill(6) for x in symbols
                        if str(x).strip().isdigit() and len(str(x).strip()) <= 6})
    if not requested:
        raise ValueError("Baidu industry request universe is empty")
    client = session or requests
    now = clock or (lambda: datetime.now(timezone.utc))
    snapshot_id = str(uuid.uuid4())
    rows = []
    failures = []

    for offset in range(0, len(requested), batch_size):
        batch = requested[offset:offset + batch_size]
        payload = [{"code": code, "market": "ab", "type": "stock"} for code in batch]
        response = client.get(
            URL,
            params={"stock": json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
                    "finClientType": "pc"},
            headers=HEADERS,
            timeout=timeout,
        )
        response.raise_for_status()
        data = response.json()
        result = data.get("Result") if isinstance(data, dict) else None
        if not isinstance(result, dict):
            raise ValueError("Baidu industry response has no Result object")
        captured = pd.Timestamp(now())
        if captured.tzinfo is None:
            raise ValueError("Baidu industry observation clock must be timezone-aware")
        captured = captured.tz_convert("UTC")

        for code in batch:
            entries = result.get(code)
            if not isinstance(entries, list):
                failures.append(code)
                continue
            found = False
            for group in entries:
                if not isinstance(group, dict) or group.get("name") != "行业":
                    continue
                for item in group.get("list", []) or []:
                    name = str(item.get("name") or "").strip()
                    if not name:
                        continue
                    query = parse_qs(str(item.get("xcx_query") or ""))
                    raw_codes = query.get("code", [])
                    industry_code = str(raw_codes[0]).strip() if raw_codes else ""
                    industry_type = str(item.get("describe") or "").strip()
                    if not industry_code:
                        industry_code = "BAIDU::" + name
                    revision = hashlib.sha256(
                        f"{code}|{industry_code}|{name}|{industry_type}".encode("utf-8")
                    ).hexdigest()
                    rows.append({
                        "symbol": code,
                        "industry_code": industry_code,
                        "industry_name": name,
                        "industry_level1": industry_type or name,
                        "industry_level2": name,
                        "snapshot_id": snapshot_id,
                        "expected_industry_count": 0,
                        "observed_at": captured,
                        "available_time": captured,
                        "availability_basis": "retrieval_time_only",
                        "historical_pit_eligible": False,
                        "source_id": "baidu.gushitong.relatedblock",
                        "snapshot_scope": "current_membership_observation_only",
                        "source_revision": revision,
                    })
                    found = True
            if not found:
                failures.append(code)
        if pause_seconds > 0 and offset + batch_size < len(requested):
            sleep_fn(pause_seconds)

    if not rows:
        raise RuntimeError(
            "Baidu industry source returned no usable classifications; "
            f"unmatched={len(failures)}/{len(requested)}"
        )
    frame = pd.DataFrame(rows)
    if frame.duplicated(["symbol", "industry_code"]).any():
        frame = frame.drop_duplicates(["symbol", "industry_code"], keep="first")
    frame["expected_industry_count"] = int(frame["industry_code"].nunique())
    frame.attrs["requested_symbol_count"] = len(requested)
    frame.attrs["unmatched_symbols"] = sorted(set(failures))
    return frame[[
        "symbol", "industry_code", "industry_name", "industry_level1",
        "industry_level2", "snapshot_id", "expected_industry_count",
        "observed_at", "available_time", "availability_basis",
        "historical_pit_eligible", "source_id", "snapshot_scope",
        "source_revision",
    ]]
