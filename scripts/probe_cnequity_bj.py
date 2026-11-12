"""Qualify CNEquity's free Beijing-board history path via full-market ingestion."""
from __future__ import annotations
import json, os
from pathlib import Path
import pandas as pd
from cnequity.query import load

config = Path("output/cnequity-bj.toml").resolve()
os.environ["CNE_CONFIG"] = str(config)
report = {
    "provider": "CNEquity",
    "target": "830799.BJ",
    "config": str(config),
    "historical_pit_eligible": False,
}
try:
    frame = load("daily_bars", symbols=["830799.BJ"], adjust=None)
    if not isinstance(frame, pd.DataFrame):
        if hasattr(frame, "to_dicts"):
            frame = pd.DataFrame(frame.to_dicts())
        else:
            raise TypeError(f"unexpected query result type: {type(frame).__name__}")
    report["rows"] = int(len(frame))
    report["columns"] = list(frame.columns)
    if len(frame):
        date_col = "trade_date" if "trade_date" in frame.columns else "date"
        report["first_date"] = str(pd.to_datetime(frame[date_col]).min().date())
        report["last_date"] = str(pd.to_datetime(frame[date_col]).max().date())
        report["symbols"] = sorted(frame["symbol"].astype(str).unique().tolist()) if "symbol" in frame else []
    required = {"open", "high", "low", "close", "volume", "amount", "symbol"}
    report["missing_fields"] = sorted(required - set(frame.columns))
    date_col = "trade_date" if "trade_date" in frame.columns else "date"
    report["unique_sessions"] = int(pd.to_datetime(frame[date_col]).nunique()) if len(frame) and date_col in frame else 0
    report["target_symbol_present"] = bool(len(frame) and set(frame["symbol"].astype(str)) == {"830799.BJ"})
    report["qualified"] = (
        report["unique_sessions"] >= 119
        and report["target_symbol_present"]
        and not report["missing_fields"]
    )
except Exception as exc:
    report["qualified"] = False
    report["error"] = f"{type(exc).__name__}: {str(exc)[:400]}"
Path("output").mkdir(exist_ok=True)
Path("output/cnequity_bj_probe.json").write_text(
    json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
)
print(json.dumps(report, ensure_ascii=False, indent=2))
if not report["qualified"]:
    raise SystemExit("CNEquity Beijing-board data qualification failed")
