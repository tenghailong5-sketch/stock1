"""Live verification that BaoStock industry date queries return dated historical snapshots."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import baostock as bs
import pandas as pd

from src.providers.baostock_industry import fetch_baostock_industry_membership_snapshot

QUERY_DATES = ["2023-12-29", "2024-12-31", "2025-12-31", "2026-11-13"]
records = []
try:
    for query_date in QUERY_DATES:
        frame = fetch_baostock_industry_membership_snapshot(
            bs, as_of_date=query_date
        )
        update_dates = pd.to_datetime(frame["updateDate"], errors="coerce")
        if update_dates.isna().any():
            raise AssertionError(f"{query_date}: provider returned invalid updateDate")
        latest_update = update_dates.max().strftime("%Y-%m-%d")
        if latest_update > query_date:
            raise AssertionError(
                f"{query_date}: provider returned future updateDate {latest_update}"
            )
        if not frame["query_date"].eq(query_date).all():
            raise AssertionError(f"{query_date}: query_date was not preserved")
        payload = frame.sort_values("symbol")[
            ["symbol", "industry_code", "industry_name", "updateDate"]
        ].to_csv(index=False)
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        records.append({
            "query_date": query_date,
            "rows": int(len(frame)),
            "unique_symbols": int(frame["symbol"].nunique()),
            "update_date_min": update_dates.min().strftime("%Y-%m-%d"),
            "update_date_max": latest_update,
            "content_sha256": digest,
            "historical_pit_eligible": bool(frame["historical_pit_eligible"].any()),
        })
finally:
    try:
        bs.logout()
    except Exception:
        pass

if len({item["content_sha256"] for item in records}) != len(QUERY_DATES):
    raise AssertionError(
        "BaoStock returned identical industry snapshots for different historical dates; "
        "date-specific history is not demonstrated."
    )
if any(item["historical_pit_eligible"] for item in records):
    raise AssertionError("Historical PIT eligibility must remain disabled.")

Path("baostock_industry_date_probe.json").write_text(
    json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
)
print(json.dumps(records, ensure_ascii=False, indent=2))
