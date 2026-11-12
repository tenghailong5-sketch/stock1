"""Test Tencent daily-history throughput and symbol coverage beyond hand-picked names."""
from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import akshare as ak
import pandas as pd

OUT = Path("output")
OUT.mkdir(exist_ok=True)
PREFIX_MARKET = {"6": "sh", "0": "sz", "3": "sz", "4": "bj", "8": "bj"}


def normalize_code(code):
    value = str(code).strip().zfill(6)
    market = PREFIX_MARKET.get(value[0])
    return f"{market}{value}" if market else None


def fetch_universe():
    """Try independent universe interfaces with bounded retries and schema normalization."""
    attempts = []
    sources = [
        ("ak.stock_info_a_code_name", lambda: ak.stock_info_a_code_name()),
        ("ak.stock_zh_a_spot_em", lambda: ak.stock_zh_a_spot_em()),
    ]
    for source_name, fetch in sources:
        for attempt in range(1, 4):
            try:
                frame = fetch()
                if frame is None or frame.empty:
                    raise ValueError("empty_response")
                if {"code", "name"}.issubset(frame.columns):
                    result = frame[["code", "name"]].copy()
                elif {"代码", "名称"}.issubset(frame.columns):
                    result = frame[["代码", "名称"]].rename(columns={"代码": "code", "名称": "name"})
                else:
                    raise ValueError(f"unexpected_columns={list(frame.columns)[:12]}")
                result["code"] = result["code"].astype(str).str.extract(r"(\\d{6})", expand=False)
                result = result.dropna(subset=["code"])
                if result.empty:
                    raise ValueError("no_six_digit_codes")
                return result, {"source": source_name, "attempt": attempt, "attempts": attempts}
            except Exception as exc:
                attempts.append({
                    "source": source_name,
                    "attempt": attempt,
                    "error": f"{type(exc).__name__}: {str(exc)[:180]}",
                })
                if attempt < 3:
                    time.sleep(2 ** (attempt - 1))
    raise RuntimeError(json.dumps({"message": "all_universe_sources_failed", "attempts": attempts}, ensure_ascii=False))


def main():
    checked = datetime.now(timezone.utc)
    end = checked.date()
    start = end - timedelta(days=220)
    report = {
        "provider": "Tencent via AKShare stock_zh_a_hist_tx",
        "checked_at_utc": checked.isoformat(),
        "requested_start": start.strftime("%Y%m%d"),
        "requested_end": end.strftime("%Y%m%d"),
        "historical_pit_eligible": False,
        "pit_note": "Retrieved historical bars are not historical point-in-time snapshots.",
    }
    try:
        universe, universe_meta = fetch_universe()
        report["universe_source"] = universe_meta
        universe["symbol"] = universe["code"].map(normalize_code)
        universe = universe.dropna(subset=["symbol"]).drop_duplicates("symbol").sort_values("symbol")
        report["universe"] = {
            "rows": int(len(universe)),
            "prefix_counts": {str(k): int(v) for k, v in universe["symbol"].str[:2].value_counts().to_dict().items()},
        }
    except Exception as exc:
        report["universe_error"] = f"{type(exc).__name__}: {str(exc)[:1000]}"
        report["qualified"] = False
        report["qualification_scope"] = "not qualified: symbol universe unavailable"
        (OUT / "tencent_60_symbol_probe.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
        raise SystemExit("Tencent universe acquisition failed after fallback retries")

    # Deterministic spread across the available universe, preserving board coverage.
    picks = []
    for prefix in ["sh", "sz", "bj"]:
        group = universe[universe["symbol"].str.startswith(prefix)]
        if len(group):
            count = min(20, len(group))
            positions = sorted(set(round(i * (len(group) - 1) / max(count - 1, 1)) for i in range(count)))
            picks.extend(group.iloc[positions]["symbol"].tolist())
    picks = list(dict.fromkeys(picks))
    report["sample_size"] = len(picks)

    items, frames = [], []
    for index, symbol in enumerate(picks, 1):
        item = {"symbol": symbol}
        try:
            frame = ak.stock_zh_a_hist_tx(
                symbol=symbol,
                start_date=report["requested_start"],
                end_date=report["requested_end"],
                adjust="",
            )
            required = {"date", "open", "close", "high", "low", "volume", "amount"}
            if frame is None or frame.empty:
                raise ValueError("empty_response")
            missing = sorted(required - set(frame.columns))
            if missing:
                raise ValueError(f"missing_columns={missing}")
            data = frame[list(required)].copy()
            data["date"] = pd.to_datetime(data["date"], errors="coerce")
            for col in ["open", "close", "high", "low", "volume", "amount"]:
                data[col] = pd.to_numeric(data[col], errors="coerce")
            if data.isna().any().any():
                raise ValueError("null_or_non_numeric_values")
            if data["date"].duplicated().any():
                raise ValueError("duplicate_dates")
            if not (data[["open", "close", "high", "low"]] > 0).all().all():
                raise ValueError("non_positive_ohlc")
            if (data["high"] < data[["open", "close", "low"]].max(axis=1)).any():
                raise ValueError("invalid_high")
            if (data["low"] > data[["open", "close", "high"]].min(axis=1)).any():
                raise ValueError("invalid_low")
            if (data[["volume", "amount"]] < 0).any().any():
                raise ValueError("negative_volume_or_amount")
            data = data.sort_values("date").reset_index(drop=True)
            data.insert(0, "symbol", symbol)
            frames.append(data)
            item.update({
                "status": "OK",
                "rows": int(len(data)),
                "first_date": data.date.min().strftime("%Y-%m-%d"),
                "last_date": data.date.max().strftime("%Y-%m-%d"),
            })
        except Exception as exc:
            item.update({"status": "ERROR", "error": f"{type(exc).__name__}: {str(exc)[:220]}"})
        items.append(item)
        print(json.dumps({"progress": f"{index}/{len(picks)}", **item}, ensure_ascii=False), flush=True)
        time.sleep(0.15)

    report["items"] = items
    report["passed_symbols"] = sum(x["status"] == "OK" for x in items)
    report["failed_symbols"] = [x for x in items if x["status"] != "OK"]
    if frames:
        combined = pd.concat(frames, ignore_index=True)
        counts = combined.groupby("symbol")["date"].nunique()
        report["total_rows"] = int(len(combined))
        report["minimum_sessions"] = int(counts.min())
        report["maximum_sessions"] = int(counts.max())
        report["symbols_with_at_least_119_sessions"] = int((counts >= 119).sum())
        path = OUT / "tencent_60_symbol_sample.csv"
        combined.to_csv(path, index=False)
        report["artifact_file"] = str(path)
        report["artifact_bytes"] = path.stat().st_size
        report["artifact_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    report["qualified"] = (
        len(picks) >= 40
        and report["passed_symbols"] >= int(len(picks) * 0.95)
        and report.get("symbols_with_at_least_119_sessions", 0) >= int(len(picks) * 0.95)
    )
    report["qualification_scope"] = "stratified sample only; full-universe bulk acquisition not yet qualified"
    (OUT / "tencent_60_symbol_probe.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if not report["qualified"]:
        raise SystemExit("Tencent multi-symbol throughput qualification failed")


if __name__ == "__main__":
    main()
