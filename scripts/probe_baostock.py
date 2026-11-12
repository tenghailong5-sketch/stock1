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
        # New BaoStock 0.9.3+ endpoint: one request returns the entire A-share
        # daily snapshot for a date. This is the critical test for avoiding
        # thousands of per-symbol requests in the production scan.
        try:
            bulk_method = getattr(bs, "query_daily_history_k_AStock")
            bulk = rows_from_result(bulk_method(date=asof))
            required_bulk = {"date", "code", "open", "high", "low", "close", "volume", "amount"}
            missing_bulk = sorted(required_bulk - set(bulk.columns))
            if not missing_bulk:
                for col in ("open", "high", "low", "close", "volume", "amount"):
                    bulk[col] = pd.to_numeric(bulk[col], errors="coerce")
                valid_bulk = bulk.dropna(subset=["date", "code", "open", "high", "low", "close", "volume", "amount"])
                valid_bulk = valid_bulk[(valid_bulk[["open", "high", "low", "close"]] > 0).all(axis=1)]
                report["bulk_daily_snapshot"] = {
                    "status": "OK" if len(valid_bulk) >= 3000 else "INSUFFICIENT_COVERAGE",
                    "rows": int(len(bulk)),
                    "valid_rows": int(len(valid_bulk)),
                    "fields": list(bulk.columns),
                    "missing_required_fields": [],
                    "sample_codes_present": sorted(set(valid_bulk["code"].astype(str)) & set(SAMPLES)),
                    "has_volume": bool(valid_bulk["volume"].notna().all()),
                    "has_amount": bool(valid_bulk["amount"].notna().all()),
                    "volume_unit": "shares",
                    "price_adjustment": "unadjusted",
                }
            else:
                report["bulk_daily_snapshot"] = {
                    "status": "SCHEMA_MISMATCH", "rows": int(len(bulk)),
                    "fields": list(bulk.columns), "missing_required_fields": missing_bulk,
                }
        except Exception as exc:
            report["bulk_daily_snapshot"] = {"status": "ERROR", "reason": str(exc)[:240]}

        end = asof
        start = (datetime.strptime(asof, "%Y-%m-%d") - timedelta(days=240)).strftime("%Y-%m-%d")
        # Reconcile query_all_stock (which also contains non-equity instruments)
        # against stock_basic.type/status before judging universe coverage.
        try:
            basic = rows_from_result(bs.query_stock_basic())
            report["stock_basic"] = {
                "rows": int(len(basic)), "fields": list(basic.columns),
                "type_counts": {str(k): int(v) for k, v in basic.get("type", pd.Series(dtype=str)).astype(str).value_counts().to_dict().items()},
                "status_counts": {str(k): int(v) for k, v in basic.get("status", pd.Series(dtype=str)).astype(str).value_counts().to_dict().items()},
            }
            active_equities = basic[(basic["type"].astype(str) == "1") & (basic["status"].astype(str) == "1")]
            report["active_equity_universe"] = {
                "rows": int(len(active_equities)),
                "exchange_prefix_counts": {str(k): int(v) for k, v in active_equities["code"].astype(str).str.extract(r"^([a-z]+)\\.", expand=False).value_counts().to_dict().items()},
                "sample_codes_present": sorted(set(active_equities["code"].astype(str)) & set(SAMPLES)),
                "codes_missing_from_bulk": sorted(set(active_equities["code"].astype(str)) - set(bulk.get("code", pd.Series(dtype=str)).astype(str)))[:100],
                "missing_from_bulk_count": int(len(set(active_equities["code"].astype(str)) - set(bulk.get("code", pd.Series(dtype=str)).astype(str)))),
            }
        except Exception as exc:
            report["stock_basic"] = {"status": "ERROR", "reason": str(exc)[:240]}

        # Verify that the bulk endpoint is historical, not just a latest-day
        # snapshot. Five dates are sampled across the last 240 calendar days.
        try:
            cal = rows_from_result(bs.query_trade_dates(start_date=start, end_date=end))
            trading_dates = cal.loc[cal["is_trading_day"].astype(str) == "1", "calendar_date"].astype(str).tolist()
            if len(trading_dates) >= 5:
                indexes = sorted(set(round(i * (len(trading_dates) - 1) / 4) for i in range(5)))
                historical_snapshots = []
                for idx in indexes:
                    day = trading_dates[idx]
                    frame = rows_from_result(bs.query_daily_history_k_AStock(date=day))
                    required = {"date", "code", "open", "high", "low", "close", "volume", "amount"}
                    missing = sorted(required - set(frame.columns))
                    historical_snapshots.append({
                        "date": day, "rows": int(len(frame)),
                        "status": "OK" if not missing and len(frame) >= 3000 else "INSUFFICIENT_OR_SCHEMA_ERROR",
                        "missing_fields": missing,
                        "sample_codes_present": sorted(set(frame.get("code", pd.Series(dtype=str)).astype(str)) & set(SAMPLES)),
                    })
                report["historical_bulk_snapshots"] = {
                    "requested_dates": len(indexes),
                    "passed": sum(x["status"] == "OK" for x in historical_snapshots),
                    "items": historical_snapshots,
                }
            else:
                report["historical_bulk_snapshots"] = {"status": "INSUFFICIENT_CALENDAR", "trading_dates": len(trading_dates)}
        except Exception as exc:
            report["historical_bulk_snapshots"] = {"status": "ERROR", "reason": str(exc)[:240]}

        # A deterministic 10-name sample tests that historical bars are not
        # merely available for the four hand-picked large-cap examples.
        try:
            if "active_equities" in locals() and len(active_equities):
                sample = active_equities.sort_values("code").iloc[::max(1, len(active_equities)//10)].head(10)
                hist_checks = []
                for code in sample["code"].astype(str):
                    try:
                        h = query_history(code, start, end)
                        hist_checks.append({"code": code, "rows": int(len(h)), "status": "OK" if len(h) >= 119 else "INSUFFICIENT_HISTORY"})
                    except Exception as exc:
                        hist_checks.append({"code": code, "status": "ERROR", "reason": str(exc)[:160]})
                report["historical_coverage_sample"] = {
                    "sample_size": len(hist_checks), "passed_119": sum(x["status"] == "OK" for x in hist_checks),
                    "failed": [x for x in hist_checks if x["status"] != "OK"],
                    "items": hist_checks,
                }
        except Exception as exc:
            report["historical_coverage_sample"] = {"status": "ERROR", "reason": str(exc)[:240]}

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
    universe = report.get("active_equity_universe", {})
    universe_ok = universe.get("rows", 0) >= 3000
    bulk = report.get("bulk_daily_snapshot", {})
    bulk_codes = set(bulk.get("sample_codes_present", []))
    bulk_ok = bulk.get("status") == "OK" and len(bulk_codes) >= 4
    missing_count = universe.get("missing_from_bulk_count", 999999)
    active_count = universe.get("rows", 0)
    bulk_coverage_ok = active_count > 0 and missing_count / active_count <= 0.02
    hist_sample = report.get("historical_coverage_sample", {})
    hist_sample_ok = hist_sample.get("sample_size", 0) >= 8 and hist_sample.get("passed_119", 0) >= 0.9 * hist_sample.get("sample_size", 1)
    historical_bulk = report.get("historical_bulk_snapshots", {})
    historical_bulk_ok = historical_bulk.get("requested_dates", 0) >= 5 and historical_bulk.get("passed", 0) == historical_bulk.get("requested_dates", -1)
    bj_ok = report.get("sample_histories", {}).get("bj.830799", {}).get("status") == "OK"
    report["qualification"] = {
        "active_equity_universe_min_3000": universe_ok,
        "single_call_full_market_daily_snapshot": bulk_ok,
        "bulk_covers_98pct_active_equities": bulk_coverage_ok,
        "historical_bulk_endpoint_5_dates": historical_bulk_ok,
        "sampled_historical_coverage_119_bars": hist_sample_ok,
        "four_market_segments_119_bars": core_ok,
        "two_major_indexes_119_bars": indexes_ok,
        "beijing_exchange_119_bars": bj_ok,
        "qualified": bool(universe_ok and bulk_ok and bulk_coverage_ok and historical_bulk_ok and hist_sample_ok and core_ok and indexes_ok and bj_ok),
    }
    target = OUT / "baostock_data_probe.json"
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    if not report["qualification"]["qualified"]:
        raise SystemExit("BaoStock data qualification failed; see output/baostock_data_probe.json")


if __name__ == "__main__":
    main()
