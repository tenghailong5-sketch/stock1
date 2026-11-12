"""Qualify Tencent daily A-share history via AKShare; sample-only, not a market-wide seed."""
from __future__ import annotations
import hashlib, json
from datetime import datetime, timedelta, timezone
from pathlib import Path
import pandas as pd
import akshare as ak

OUT = Path("output")
OUT.mkdir(exist_ok=True)
SAMPLES = ["sh600000", "sz000001", "sz300750", "sh688981"]

def main():
    end = datetime.now(timezone.utc).date()
    start = end - timedelta(days=220)
    report = {
        "provider": "Tencent via AKShare stock_zh_a_hist_tx",
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "start_date": start.strftime("%Y%m%d"),
        "end_date": end.strftime("%Y%m%d"),
        "symbols_required": SAMPLES,
        "historical_pit_eligible": False,
        "pit_note": "Historical bars fetched now are retrieval-time observations, not historical PIT snapshots.",
        "symbols": {},
        "qualified": False,
    }
    frames = []
    for symbol in SAMPLES:
        item = {"symbol": symbol}
        try:
            frame = ak.stock_zh_a_hist_tx(
                symbol=symbol,
                start_date=report["start_date"],
                end_date=report["end_date"],
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
                raise ValueError("duplicate_trade_dates")
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
                "sessions": int(data["date"].nunique()),
                "first_date": data["date"].min().strftime("%Y-%m-%d"),
                "last_date": data["date"].max().strftime("%Y-%m-%d"),
                "sha256": hashlib.sha256(data.to_csv(index=False).encode()).hexdigest(),
            })
        except Exception as exc:
            item.update({"status": "ERROR", "error": f"{type(exc).__name__}: {str(exc)[:300]}"})
        report["symbols"][symbol] = item
        print(json.dumps(item, ensure_ascii=False), flush=True)

    if len(frames) == len(SAMPLES):
        combined = pd.concat(frames, ignore_index=True)
        date_sets = [set(x["date"].dt.strftime("%Y-%m-%d")) for x in frames]
        common_sessions = set.intersection(*date_sets)
        report["total_rows"] = int(len(combined))
        report["common_sessions"] = len(common_sessions)
        report["first_common_session"] = min(common_sessions) if common_sessions else None
        report["last_common_session"] = max(common_sessions) if common_sessions else None
        report["qualified"] = (
            all(report["symbols"][s]["sessions"] >= 100 for s in SAMPLES)
            and len(common_sessions) >= 100
        )
        if report["qualified"]:
            path = OUT / "tencent_sample_daily_bars.csv"
            combined.to_csv(path, index=False)
            report["artifact_file"] = str(path)
            report["artifact_bytes"] = path.stat().st_size
            report["artifact_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    report["qualification_scope"] = "four-symbol connectivity and historical-depth sample only; not full-market coverage"
    (OUT / "tencent_provider_probe.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if not report["qualified"]:
        raise SystemExit("Tencent provider sample qualification failed")

if __name__ == "__main__":
    main()
