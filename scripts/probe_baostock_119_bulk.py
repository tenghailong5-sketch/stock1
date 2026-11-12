"""Measure a complete rolling 119-session BaoStock bulk-history seed.

This is an acquisition/coverage test only. It does not create historical PIT
snapshots or authorize formal model signals.
"""
from __future__ import annotations

import hashlib
import json
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import baostock as bs
import pandas as pd

OUT = Path("output")
OUT.mkdir(exist_ok=True)
REQUIRED = {"date", "code", "open", "high", "low", "close", "volume", "amount"}
SAMPLE_CODES = {"sh.600000", "sz.000001", "sz.300750", "sh.688981"}


def read_result(rs):
    if rs.error_code != "0":
        raise RuntimeError(f"BaoStock error {rs.error_code}: {rs.error_msg}")
    rows = []
    while rs.next():
        rows.append(rs.get_row_data())
    return pd.DataFrame(rows, columns=rs.fields)


def main():
    report = {
        "provider": "BaoStock",
        "test": "119_session_full_market_bulk_history",
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "historical_pit_eligible": False,
        "pit_note": "Historical bars fetched now are not historical publication-time snapshots.",
        "requested_sessions": 119,
        "passed_sessions": 0,
        "failed_sessions": [],
        "session_rows": [],
        "sample_codes_required_each_session": sorted(SAMPLE_CODES),
        "sample_codes_missing_by_session": [],
    }
    login = bs.login()
    if login.error_code != "0":
        raise RuntimeError(f"BaoStock login failed: {login.error_code} {login.error_msg}")
    try:
        end = date.today()
        start = end - timedelta(days=300)
        calendar = read_result(bs.query_trade_dates(
            start_date=start.isoformat(), end_date=end.isoformat()
        ))
        trading = calendar.loc[
            calendar["is_trading_day"].astype(str) == "1", "calendar_date"
        ].astype(str).tolist()
        sessions = trading[-119:]
        if len(sessions) != 119:
            raise RuntimeError(f"Need 119 trading sessions, got {len(sessions)}")
        report["first_session"] = sessions[0]
        report["last_session"] = sessions[-1]
        report["calendar_sessions_found"] = len(trading)

        frames = []
        total_started = time.monotonic()
        for i, day in enumerate(sessions, 1):
            started = time.monotonic()
            try:
                frame = read_result(bs.query_daily_history_k_AStock(date=day))
                missing = sorted(REQUIRED - set(frame.columns))
                if missing:
                    raise RuntimeError(f"missing_fields={missing}")
                if frame.empty:
                    raise RuntimeError("empty_snapshot")
                if frame["date"].astype(str).nunique() != 1 or str(frame["date"].iloc[0]) != day:
                    raise RuntimeError("returned_date_mismatch")
                if frame["code"].astype(str).duplicated().any():
                    raise RuntimeError("duplicate_codes")
                for col in ("open", "high", "low", "close", "volume", "amount"):
                    frame[col] = pd.to_numeric(frame[col], errors="coerce")
                valid = frame.dropna(subset=list(REQUIRED - {"date", "code"}))
                if len(valid) < 4500:
                    raise RuntimeError(f"low_valid_coverage={len(valid)}")
                if (valid[["open", "high", "low", "close"]] <= 0).any().any():
                    raise RuntimeError("non_positive_ohlc")
                if (valid["high"] < valid[["open", "low", "close"]].max(axis=1)).any():
                    raise RuntimeError("invalid_high")
                if (valid["low"] > valid[["open", "high", "close"]].min(axis=1)).any():
                    raise RuntimeError("invalid_low")
                present_samples = sorted(set(frame["code"].astype(str)) & SAMPLE_CODES)
                missing_samples = sorted(SAMPLE_CODES - set(present_samples))
                if missing_samples:
                    raise RuntimeError(f"missing_segment_samples={missing_samples}")
                frame["date"] = day
                frames.append(frame)
                item = {
                    "date": day,
                    "rows": int(len(frame)),
                    "valid_rows": int(len(valid)),
                    "sample_codes_present": present_samples,
                    "elapsed_seconds": round(time.monotonic() - started, 2),
                    "status": "OK",
                }
                report["passed_sessions"] += 1
            except Exception as exc:
                item = {
                    "date": day,
                    "elapsed_seconds": round(time.monotonic() - started, 2),
                    "status": "ERROR",
                    "reason": f"{type(exc).__name__}: {str(exc)[:240]}",
                }
                report["failed_sessions"].append(item)
                if "missing_segment_samples" in item.get("reason", ""):
                    report["sample_codes_missing_by_session"].append(item["date"])
            report["session_rows"].append(item)
            print(json.dumps({"progress": f"{i}/119", **item}, ensure_ascii=False), flush=True)

        report["elapsed_total_seconds"] = round(time.monotonic() - total_started, 2)
        report["total_rows"] = int(sum(len(x) for x in frames))
        report["minimum_rows_per_session"] = min(
            (x["rows"] for x in report["session_rows"] if x["status"] == "OK"),
            default=0,
        )
        report["maximum_rows_per_session"] = max(
            (x["rows"] for x in report["session_rows"] if x["status"] == "OK"),
            default=0,
        )
        report["four_market_samples_seen"] = sorted(
            set().union(*(set(x.get("sample_codes_present", [])) for x in report["session_rows"]))
        )
        if frames:
            combined = pd.concat(frames, ignore_index=True)
            combined = combined.sort_values(["date", "code"]).reset_index(drop=True)
            target = OUT / "baostock_119_sessions.parquet"
            combined.to_parquet(target, index=False, compression="zstd")
            payload = target.read_bytes()
            report["artifact_file"] = str(target)
            report["artifact_bytes"] = len(payload)
            report["artifact_sha256"] = hashlib.sha256(payload).hexdigest()
        report["qualified"] = (
            report["passed_sessions"] == 119
            and len(report["failed_sessions"]) == 0
            and report["total_rows"] >= 119 * 4500
            and SAMPLE_CODES.issubset(set(report["four_market_samples_seen"]))
            and len(report["sample_codes_missing_by_session"]) == 0
        )
        report["qualification_scope"] = "SH/SZ A-share full-market bulk history; BJ not covered by BaoStock"
    finally:
        bs.logout()

    (OUT / "baostock_119_bulk_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if not report.get("qualified"):
        raise SystemExit("119-session full-market bulk qualification failed")


if __name__ == "__main__":
    main()
