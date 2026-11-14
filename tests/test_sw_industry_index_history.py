import pandas as pd
import pytest

from src.providers.sw_industry import fetch_sw_industry_index_history


def test_fetch_sw_industry_index_history_stamps_retrieval_time_and_returns():
    raw = pd.DataFrame({
        "日期": ["2026-11-10", "2026-11-11", "2026-11-12", "2026-11-13",
                "2026-11-16", "2026-11-17"],
        "开盘": [100, 101, 102, 103, 104, 105],
        "最高": [101, 102, 103, 104, 105, 106],
        "最低": [99, 100, 101, 102, 103, 104],
        "收盘": [100, 101, 102, 103, 104, 110],
        "成交量": [1, 1, 1, 1, 1, 1],
        "成交额": [10, 10, 10, 10, 10, 10],
    })
    result = fetch_sw_industry_index_history(
        "801010",
        clock=lambda: pd.Timestamp("2026-11-18T00:00:00Z"),
        history_fetcher=lambda **kwargs: raw,
    )
    assert len(result) == 6
    assert result["industry_code"].eq("801010").all()
    assert result["historical_pit_eligible"].eq(False).all()
    assert result["availability_basis"].eq("retrieval_time_only").all()
    assert result["available_time"].eq(pd.Timestamp("2026-11-18T00:00:00Z")).all()
    assert result.iloc[-1]["return_5"] == pytest.approx(0.10)
    assert result.iloc[-1]["daily_return"] == pytest.approx(110 / 104 - 1)
    assert result["source_revision"].str.len().eq(64).all()


def test_fetch_sw_industry_index_history_rejects_invalid_code():
    with pytest.raises(ValueError, match="six-digit"):
        fetch_sw_industry_index_history("80101", history_fetcher=lambda **kwargs: None)


def test_fetch_sw_industry_index_history_rejects_duplicate_dates():
    raw = pd.DataFrame({"日期": ["2026-11-10", "2026-11-10"], "收盘": [100, 101]})
    with pytest.raises(ValueError, match="duplicate dates"):
        fetch_sw_industry_index_history(
            "801010", history_fetcher=lambda **kwargs: raw
        )
