"""Qualify Tencent history using exchange-native A-share security masters."""
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
CODE_COLUMNS = ("证券代码", "A股代码", "股票代码", "代码", "证券代码 ")
NAME_COLUMNS = ("证券简称", "A股简称", "股票简称", "名称", "证券名称")


def normalize_code(code):
    digits = "".join(ch for ch in str(code).strip() if ch.isdigit())
    if not digits:
        return None
    value = digits[-6:].zfill(6)
    market = PREFIX_MARKET.get(value[0])
    return f"{market}{value}" if market else None


def normalize_exchange_frame(frame, source):
    if frame is None or frame.empty:
        raise ValueError(f"{source}: empty_response")
    code_col = next((c for c in CODE_COLUMNS if c in frame.columns), None)
    name_col = next((c for c in NAME_COLUMNS if c in frame.columns), None)
    if not code_col:
        raise ValueError(f"{source}: no_code_column; columns={list(frame.columns)[:15]}")
    out = pd.DataFrame({"raw_code": frame[code_col].astype(str)})
    out["name"] = frame[name_col].astype(str) if name_col else ""
    out["symbol"] = out["raw_code"].map(normalize_code)
    out = out.dropna(subset=["symbol"]).drop_duplicates("symbol")
    if out.empty:
        raise ValueError(f"{source}: no_valid_A_share_codes")
    out["source"] = source
    return out[["symbol", "name", "source"]]


def retry(label, fn, attempts=3):
    errors = []
    for attempt in range(1, attempts + 1):
        try:
            return fn(), {"source": label, "attempt": attempt, "previous_errors": errors}
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {str(exc)[:180]}")
            if attempt < attempts:
                time.sleep(min(2 ** (attempt - 1), 4))
    raise RuntimeError(f"{label} failed: {' | '.join(errors)}")


def fetch_exchange_universe():
    # These three interfaces are exchange-specific; deliberately avoid the
    # Eastmoney Push2 endpoints that were rate-limited in the previous run.
    tasks = [
        ("SSE main-board A shares", lambda: ak.stock_info_sh_name_code(indicator="主板A股")),
        ("SSE STAR Market", lambda: ak.stock_info_sh_name_code(indicator="科创板")),
        ("SZSE A shares", lambda: ak.stock_info_sz_name_code(symbol="A股列表")),
        ("BSE listed shares", lambda: ak.stock_info_bj_name_code()),
    ]
    frames, diagnostics = [], []
    for label, fn in tasks:
        raw, meta = retry(label, fn)
        normalized = normalize_exchange_frame(raw, label)
        diagnostics.append({**meta, "rows": int(len(normalized))})
        frames.append(normalized)
    universe = pd.concat(frames, ignore_index=True).drop_duplicates("symbol").sort_values("symbol")
    if len(universe) < 3000:
        raise RuntimeError(f"exchange universe unexpectedly small: {len(universe)}; details={diagnostics}")
    return universe.reset_index(drop=True), diagnostics


def main():
    checked = datetime.now(timezone.utc)
    end = checked.date()
    start = end - timedelta(days=220)
    report = {
        "provider": "Tencent via AKShare stock_zh_a_hist_tx",
        "universe_provider": "SSE + SZSE + BSE exchange-specific security lists via AKShare",
        "checked_at_utc": checked.isoformat(),
        "requested_start": start.strftime("%Y%m%d"),
        "requested_end": end.strftime("%Y%m%d"),
        "historical_pit_eligible": False,
        "pit_note": "Historical bars fetched now are not historical point-in-time snapshots.",
    }
    try:
        universe, diagnostics = fetch_exchange_universe()
        report["universe_sources"] = diagnostics
        report["universe"] = {
            "rows": int(len(universe)),
            "exchange_counts": {
                prefix: int(universe["symbol"].str.startswith(prefix).sum())
                for prefix in ("sh", "sz", "bj")
            },
        }
    except Exception as exc:
        report["universe_error"] = f"{type(exc).__name__}: {str(exc)[:1500]}"
        report["qualified"] = False
        report["qualification_scope"] = "not qualified: exchange-native symbol universe unavailable"
        (OUT / "tencent_60_symbol_probe.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
        raise SystemExit("Exchange-native universe acquisition failed")

    # Deterministic stratified sample across all three exchanges.
    picks = []
    for prefix in ("sh", "sz", "bj"):
        group = universe[universe["symbol"].str.startswith(prefix)]
        count = min(20, len(group))
        if count:
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
            for col in ("open", "close", "high", "low", "volume", "amount"):
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
                "first_date": data["date"].min().strftime("%Y-%m-%d"),
                "last_date": data["date"].max().strftime("%Y-%m-%d"),
            })
        except Exception as exc:
            item.update({"status": "ERROR", "error": f"{type(exc).__name__}: {str(exc)[:220]}"})
        items.append(item)
        print(json.dumps({"progress": f"{index}/{len(picks)}", **item}, ensure_ascii=False), flush=True)
        time.sleep(0.2)

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
    report["qualification_scope"] = "stratified sample only; full-market historical backfill not yet qualified"
    (OUT / "tencent_60_symbol_probe.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if not report["qualified"]:
        raise SystemExit("Tencent multi-symbol history qualification failed")


if __name__ == "__main__":
    main()
