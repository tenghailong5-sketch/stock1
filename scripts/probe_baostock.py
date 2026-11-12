"""Live BaoStock data-source qualification probe.

This is a source qualification test, not a trading scan. It records actual
coverage and refuses to treat retrieved history as historical PIT data.
"""
from __future__ import annotations

import json
import os
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import baostock as bs
import pandas as pd

OUT = Path("output")
OUT.mkdir(exist_ok=True)
FIELDS = "date,code,open,high,low,close,preclose,volume,amount,adjustflag,turn,tradestatus,pctChg,isST"
SAMPLES = {
    "sh.600000": "Shanghai main board",
    "sz.000001": "Shenzhen main board",
    "sz.300750": "ChiNext",
    "sh.688981": "STAR",
    "bj.830799": "Beijing optional coverage",
}
INDEXES = {"sh.000001": "SSE Composite", "sz.399001": "SZSE Component"}


def rows_from_result(rs):
    if rs.error_code != "0":
        raise RuntimeError(f"BaoStock query error {rs.error_code}: {rs.error_msg}")
    rows = []
    while rs.next():
        rows.append(rs.get_row_data())
    return pd.DataFrame(rows, columns=rs.fields)


def query_stock_list():
    today = date.today()
    failures = []
    for offset in range(0, 8):
        day = (today - timedelta(days=offset)).isoformat()
        try:
            frame = rows_from_result(bs.query_all_stock(day=day))
            if not frame.empty:
                return day, frame, failures
            failures.append({"date": day, "reason": "empty_stock_list"})
        except Exception as exc:
            failures.append({"date": day, "reason": str(exc)[:180]})
    raise RuntimeError(f"BaoStock returned no stock list in last 8 days: {failures}")


def query_history(code, start, end):
    rs = bs.query_history_k_data_plus(
        code, FIELDS, start_date=start, end_date=end,
        frequency="d", adjustflag="3"
    )
    frame = rows_from_result(rs)
    if frame.empty:
        raise RuntimeError("empty_daily_history")
    for col in ("open", "high", "low", "close", "volume", "amount"):
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.dropna(subset=["date", "open", "high", "low", "close"])
    if frame["date"].duplicated().any():
        raise RuntimeError("duplicate_daily_dates")
    if (frame[["open", "high", "low", "close"]] <= 0).any().any():
        raise RuntimeError("non_positive_ohlc")
    if (frame["high"] < frame[["open", "low", "close"]].max(axis=1)).any():
        raise RuntimeError("invalid_high")
    if (frame["low"] > frame[["open", "high", "close"]].min(axis=1)).any():
        raise RuntimeError("invalid_low")
    return frame.sort_values("date")


def main():
    report = {
        "provider": "BaoStock",
        "package_version": getattr(bs, "__version__", "unknown"),
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "historical_pit_eligible": False,
        "pit_note": "Historical bars fetched now are not historical publication-time snapshots.",
        "stock_universe": {},
        "sample_histories": {},
        "index_histories": {},
        "errors": [],
    }
    login = bs.login()
    if login.error_code != "0":
        raise RuntimeError(f"BaoStock login failed: {login.error_code} {login.error_msg}")
    try:
        asof, stocks, list_retries = query_stock_list()
        report["stock_universe"] = {
            "asof": asof,
            "rows": int(len(stocks)),
            "fields": list(stocks.columns),
            "sample_codes_present": sorted(set(stocks["code"].astype(str)) & set(SAMPLES)),
            "list_retries": list_retries,
            "market_code_prefix_counts": {
                str(k): int(v) for k, v in
                stocks["code"].astype(str).str.extract(r"^([a-z]+)\.", expand=False)
                .value_counts().to_dict().items()
            },
        }
        end = asof
        start = (datetime.strptime(asof, "%Y-%m-%d") - timedelta(days=240)).strftime("%Y-%m-%d")
        present = set(stocks["code"].astype(str))
        for code, market in SAMPLES.items():
            if code not in present:
                report["sample_histories"][code] = {
                    "market": market, "status": "NOT_IN_UNIVERSE", "rows": 0
                }
                continue
            try:
                frame = query_history(code, start, end)
                report["sample_histories"][code] = {
                    "market": market, "status": "OK" if len(frame) >= 119 else "INSUFFICIENT_HISTORY",
                    "rows": int(len(frame)),
                    "first_date": frame["date"].min().date().isoformat(),
                    "last_date": frame["date"].max().date().isoformat(),
                    "has_volume": bool(frame["volume"].notna().all()),
                    "has_amount": bool(frame["amount"].notna().all()),
                    "volume_unit": "shares",
                    "price_adjustment": "unadjusted",
                }
            except Exception as exc:
                report["sample_histories"][code] = {
                    "market": market, "status": "ERROR", "reason": str(exc)[:240]
                }
        index_fields = "date,code,open,high,low,close,volume,amount"
        for code, name in INDEXES.items():
            try:
                rs = bs.query_history_k_data_plus(
                    code, index_fields, start_date=start, end_date=end,
                    frequency="d", adjustflag="3"
                )
                frame = rows_from_result(rs)
                frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
                report["index_histories"][code] = {
                    "name": name, "status": "OK" if len(frame) >= 119 else "INSUFFICIENT_HISTORY",
                    "rows": int(len(frame)),
                    "first_date": frame["date"].min().date().isoformat() if len(frame) else None,
                    "last_date": frame["date"].max().date().isoformat() if len(frame) else None,
                }
            except Exception as exc:
                report["index_histories"][code] = {
                    "name": name, "status": "ERROR", "reason": str(exc)[:240]
                }
    finally:
        bs.logout()

    core_stocks = ["sh.600000", "sz.000001", "sz.300750", "sh.688981"]
    core_ok = all(report["sample_histories"].get(code, {}).get("status") == "OK" for code in core_stocks)
    indexes_ok = all(report["index_histories"].get(code, {}).get("status") == "OK" for code in INDEXES)
    universe_ok = report["stock_universe"].get("rows", 0) >= 3000
    report["qualification"] = {
        "universe_min_3000": universe_ok,
        "four_market_segments_119_bars": core_ok,
        "two_major_indexes_119_bars": indexes_ok,
        "qualified": bool(universe_ok and core_ok and indexes_ok),
    }
    target = OUT / "baostock_data_probe.json"
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    if not report["qualification"]["qualified"]:
        raise SystemExit("BaoStock data qualification failed; see output/baostock_data_probe.json")


if __name__ == "__main__":
    main()
