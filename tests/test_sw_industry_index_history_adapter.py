import pandas as pd
import pytest

from src.providers.sw_industry import fetch_sw_industry_index_history


def test_sw_industry_index_history_normalizes_realistic_provider_schema():
    raw = pd.DataFrame({
        "index_code": ["801010"] * 7,
        "index_name": ["农林牧渔"] * 7,
        "date": pd.date_range("2026-11-02", periods=7).strftime("%Y-%m-%d"),
        "open": [100, 101, 102, 103, 104, 105, 106],
        "high": [101, 102, 103, 104, 105, 106, 107],
        "low": [99, 100, 101, 102, 103, 104, 105],
        "close": [100, 101, 102, 103, 104, 105, 110],
        "vol": [1000] * 7,
        "amount": [1000000, 1100000, 1200000, 1300000, 1400000, 1500000, 1800000],
    })
    result = fetch_sw_industry_index_history(
        "801010", start_date="20261102", end_date="20261108",
        clock=lambda: pd.Timestamp("2026-11-09T00:00:00Z"),
        history_fetcher=lambda **kwargs: raw,
    )
    assert len(result) == 7
    assert result.industry_code.eq("801010").all()
    assert result.historical_pit_eligible.eq(False).all()
    assert result.availability_basis.eq("retrieval_time_only").all()
    assert result.iloc[-1].return_5 == pytest.approx(0.10)
    assert result.iloc[-1].daily_return == pytest.approx(110 / 105 - 1)
    assert result.iloc[-1].amount == 1800000


def test_sw_industry_index_history_rejects_invalid_date_and_code():
    with pytest.raises(ValueError, match="six-digit"):
        fetch_sw_industry_index_history(
            "80101", start_date="20261101", end_date="20261110",
            history_fetcher=lambda **kwargs: None,
        )
    with pytest.raises(ValueError, match="YYYYMMDD"):
        fetch_sw_industry_index_history(
            "801010", start_date="2026-11-01", end_date="20261110",
            history_fetcher=lambda **kwargs: None,
        )


def test_sw_industry_index_history_rejects_missing_amount_and_duplicate_dates():
    missing_amount = pd.DataFrame({"date": ["2026-11-02"], "close": [100]})
    with pytest.raises(ValueError, match="missing columns"):
        fetch_sw_industry_index_history(
            "801010", start_date="20261101", end_date="20261110",
            history_fetcher=lambda **kwargs: missing_amount,
        )
    duplicate_dates = pd.DataFrame({
        "date": ["2026-11-02", "2026-11-02"],
        "close": [100, 101], "amount": [1000, 1100],
    })
    with pytest.raises(ValueError, match="duplicate dates"):
        fetch_sw_industry_index_history(
            "801010", start_date="20261101", end_date="20261110",
            history_fetcher=lambda **kwargs: duplicate_dates,
        )
