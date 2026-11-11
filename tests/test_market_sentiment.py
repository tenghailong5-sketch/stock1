import pandas as pd
import pytest

from src.model.market_sentiment import summarize_limit_pool_sentiment


def _snapshots():
    rows = []
    pools = {
        "limit_up": [{"代码": "000001", "连板数": 3}, {"代码": "000002", "连板数": None}, {"代码": "000003", "连板数": 2}],
        "limit_down": [],
        "limit_up_opened": [{"代码": "000004", "炸板次数": 1}],
        "previous_limit_up": [{"代码": "000005"}],
        "strong": [{"代码": "000006"}],
    }
    for event_type, records in pools.items():
        rows.append({"event_type": event_type, "provider_row_count": len(records), "records": records})
    return pd.DataFrame(rows)


def test_summary_reports_observed_counts_and_ladder_height_only():
    result = summarize_limit_pool_sentiment(_snapshots())
    assert result["limit_up_count"] == 3
    assert result["limit_down_count"] == 0
    assert result["limit_up_opened_pool_count"] == 1
    assert result["highest_observed_limit_up_streak"] == 3
    assert result["limit_up_streak_observed_count"] == 2
    assert result["limit_up_streak_missing_count"] == 1
    assert result["opened_rate"] is None
    assert result["scoring_eligible"] is False
    assert result["historical_pit_eligible"] is False


def test_missing_pool_fails_closed():
    frame = _snapshots().query("event_type != 'strong'")
    with pytest.raises(ValueError, match="missing required"):
        summarize_limit_pool_sentiment(frame)


def test_duplicate_pool_fails_closed():
    frame = pd.concat([_snapshots(), _snapshots().iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        summarize_limit_pool_sentiment(frame)


def test_count_records_mismatch_fails_closed():
    frame = _snapshots()
    frame.loc[frame["event_type"] == "limit_up", "provider_row_count"] = 99
    with pytest.raises(ValueError, match="mismatch"):
        summarize_limit_pool_sentiment(frame)
