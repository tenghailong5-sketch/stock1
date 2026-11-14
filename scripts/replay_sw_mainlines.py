"""Retrospective SW first-level industry mainline replay from official index history."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import requests

from src.model.mainline_flows import classify_mainlines
from src.providers.sw_api_industry import (
    fetch_sw_api_first_level_index_list,
    fetch_sw_api_industry_index_history,
)


def replay(start_date: str, end_date: str, *, output_dir: Path = Path("output")) -> dict:
    session = requests.Session()
    groups = fetch_sw_api_first_level_index_list(session=session)
    expected = len(groups)
    if expected < 20:
        raise RuntimeError(f"SW first-level industry list unexpectedly small: {expected}")

    frames, failures = [], []
    for row in groups.itertuples(index=False):
        code, name = row.industry_code, row.industry_name
        try:
            frame = fetch_sw_api_industry_index_history(
                code, start_date=start_date, end_date=end_date,
                session=session,
            )
            frame["industry_name"] = str(name)
            frames.append(frame)
            time.sleep(0.2)
        except Exception as exc:
            failures.append(f"{code}:{type(exc).__name__}:{str(exc)[:180]}")
    if failures:
        raise RuntimeError(
            f"industry history incomplete: {len(failures)}/{expected}; "
            + " | ".join(failures[:12])
        )
    panel = pd.concat(frames, ignore_index=True)
    counts = panel.groupby("trade_date")["industry_code"].nunique()
    complete_days = counts[counts == expected].index
    panel = panel[panel.trade_date.isin(complete_days)].copy()
    if panel.empty:
        raise RuntimeError("no dates have complete first-level industry index coverage")

    market_path = Path("data/forward_archive/index_daily/000001.jsonl")
    if not market_path.exists():
        raise RuntimeError("Shanghai Composite daily archive is missing")
    market_rows = [
        json.loads(line) for line in market_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    market = pd.DataFrame(market_rows)
    market["trade_date"] = pd.to_datetime(market["trade_date"], errors="coerce").dt.normalize()
    market["close"] = pd.to_numeric(market["close"], errors="coerce")
    market = market.dropna(subset=["trade_date", "close"]).sort_values("trade_date")
    market["market_return_5d"] = market["close"].div(market["close"].shift(5)).sub(1)

    panel["trade_date"] = pd.to_datetime(panel["trade_date"], errors="coerce").dt.normalize()
    panel["industry_return_5d"] = pd.to_numeric(panel["return_5"], errors="coerce")
    panel["turnover"] = pd.to_numeric(panel["amount"], errors="coerce")
    daily_total = panel.groupby("trade_date")["turnover"].sum().rename("market_turnover")
    panel = panel.merge(daily_total, on="trade_date", how="left", validate="many_to_one")
    panel = panel.merge(
        market[["trade_date", "market_return_5d"]],
        on="trade_date", how="left", validate="many_to_one",
    )
    panel = panel.dropna(subset=["industry_return_5d", "market_return_5d", "turnover", "market_turnover"])
    panel["expected_industry_count"] = expected
    panel["historical_pit_eligible"] = False
    if panel.empty:
        raise RuntimeError("no aligned industry/market history remains after date joins")

    phases = classify_mainlines(panel[[
        "trade_date", "industry_code", "industry_return_5d", "market_return_5d",
        "turnover", "market_turnover", "expected_industry_count",
    ]])
    phases["historical_pit_eligible"] = False
    phases["availability_basis"] = "retrieval_time_only"
    phases["source_id"] = "swsresearch.sw_index_daily_via_akshare"
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "sw_mainline_phase_replay.csv"
    phases.to_csv(csv_path, index=False)
    counts = phases["mainline_phase"].value_counts().to_dict()
    summary = {
        "status": "RETROSPECTIVE_REPLAY_COMPLETE",
        "industry_count": expected,
        "complete_trade_dates": int(phases.trade_date.nunique()),
        "rows": int(len(phases)),
        "phase_counts": {str(k): int(v) for k, v in counts.items()},
        "start_date": str(phases.trade_date.min().date()),
        "end_date": str(phases.trade_date.max().date()),
        "historical_pit_eligible": False,
        "output": str(csv_path),
    }
    (output_dir / "sw_mainline_phase_replay_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-date", default="20260522")
    parser.add_argument("--end-date", default="20261113")
    args = parser.parse_args()
    replay(args.start_date, args.end_date)
