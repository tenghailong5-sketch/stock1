import json

import pandas as pd

from src.model.m3_assembler import assemble_m3_history


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def make_archive(root, *, omit_breadth_day=None):
    dates = pd.bdate_range("2026-07-01", periods=80)
    index_rows = []
    for i, day in enumerate(dates):
        day_text = day.strftime("%Y-%m-%d")
        index_rows.append({
            "index_symbol": "000001",
            "trade_date": day_text,
            "close": 3000 + i * 2,
            "available_time": f"{day_text}T08:30:00Z",
            "historical_pit_eligible": False,
        })
        if day_text == omit_breadth_day:
            continue
        write_jsonl(root / "market_breadth" / f"{day_text}.jsonl", [{
            "snapshot_time": f"{day_text}T08:30:00Z",
            "breadth": 0.65,
            "breadth_coverage": 0.99,
            "total_amount": 1_000_000_000_000,
            "total_amount_status": "complete_provider_snapshot_sum",
            "historical_pit_eligible": False,
            "snapshot_scope": "current_provider_snapshot_only",
        }])
        write_jsonl(root / "market_sentiment" / f"{day_text}.jsonl", [{
            "trade_date": day_text,
            "capture_time": f"{day_text}T08:30:00Z",
            "limit_up_count": 30,
            "limit_down_count": 5,
            "historical_pit_eligible": False,
            "scoring_eligible": False,
        }])
    write_jsonl(root / "index_daily" / "000001.jsonl", index_rows)
    return dates


def test_m3_assembler_builds_state_only_after_60_complete_sessions(tmp_path):
    dates = make_archive(tmp_path)
    result = assemble_m3_history(
        tmp_path, index_symbol="000001", decision_time="2026-12-31T12:00:00Z"
    )
    assert result["status"] == "READY"
    assert result["valid_sessions"] == 80
    assert result["latest_trade_date"] == dates[-1].strftime("%Y-%m-%d")
    assert result["market_state"] == "RISK_ON"
    assert result["historical_backtest_eligible"] is False


def test_m3_assembler_fails_closed_when_one_recent_daily_snapshot_is_missing(tmp_path):
    dates = pd.bdate_range("2026-07-01", periods=80)
    missing_day = dates[-10].strftime("%Y-%m-%d")
    make_archive(tmp_path, omit_breadth_day=missing_day)
    result = assemble_m3_history(
        tmp_path, index_symbol="000001", decision_time="2026-12-31T12:00:00Z"
    )
    assert result["status"] == "INSUFFICIENT_DATA"
    assert result["reason"] == "fewer_than_60_complete_consecutive_sessions"
    assert missing_day in result["missing_recent_dates"]


def test_m3_assembler_rejects_decision_before_archive_capture(tmp_path):
    make_archive(tmp_path)
    result = assemble_m3_history(
        tmp_path, index_symbol="000001", decision_time="2026-07-01T08:00:00Z"
    )
    assert result["status"] == "INSUFFICIENT_DATA"
