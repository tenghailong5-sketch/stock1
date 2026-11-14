"""Live verification that BaoStock industry date queries return dated historical snapshots."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import baostock as bs
import pandas as pd

QUERY_DATES = ["2023-12-29", "2024-12-31", "2025-12-31", "2026-11-13"]
records = []

def retry_call(label, call, attempts=3):
    last_error = None
    for attempt in range(1, attempts + 1):
        try:
            response = call()
            if str(getattr(response, "error_code", "")) == "0":
                return response
            last_error = getattr(response, "error_msg", "unknown provider error")
        except Exception as exc:  # network disconnects can raise from the socket layer
            last_error = repr(exc)
        if attempt < attempts:
            print(f"{label}: attempt {attempt}/{attempts} failed ({last_error}); retrying")
            time.sleep(10 * attempt)
    raise RuntimeError(f"{label} failed after {attempts} attempts: {last_error}")

login = retry_call("BaoStock login", bs.login)

try:
    for query_date in QUERY_DATES:
        result = retry_call(f"{query_date} query_stock_industry", lambda: bs.query_stock_industry(date=query_date))
        rows = []
        while result.next():
            rows.append(result.get_row_data())
        frame = pd.DataFrame(rows, columns=result.fields)
        required = {"updateDate", "code", "industry", "industryClassification"}
        if frame.empty or not required.issubset(frame.columns):
            raise AssertionError(f"{query_date}: empty or invalid provider response")
        update_dates = pd.to_datetime(frame["updateDate"], errors="coerce")
        if update_dates.isna().any():
            raise AssertionError(f"{query_date}: provider returned invalid updateDate")
        latest_update = update_dates.max().strftime("%Y-%m-%d")
        if latest_update > query_date:
            raise AssertionError(
                f"{query_date}: provider returned future updateDate {latest_update}"
            )
        payload = frame.sort_values("code")[
            ["code", "industry", "industryClassification", "updateDate"]
        ].to_csv(index=False)
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        records.append({
            "query_date": query_date,
            "rows": int(len(frame)),
            "unique_symbols": int(frame["code"].nunique()),
            "update_date_min": update_dates.min().strftime("%Y-%m-%d"),
            "update_date_max": latest_update,
            "content_sha256": digest,
        })
finally:
    bs.logout()

if len({item["content_sha256"] for item in records}) != len(QUERY_DATES):
    raise AssertionError(
        "BaoStock returned identical industry snapshots for different historical dates; "
        "date-specific history is not demonstrated."
    )

Path("baostock_industry_date_probe.json").write_text(
    json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
)
print(json.dumps(records, ensure_ascii=False, indent=2))
