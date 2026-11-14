import pandas as pd
import pytest

from src.model.m4_archive_adapter import (
    build_industry_index_returns,
    build_market_index_returns,
    build_stock_daily_returns,
)


def _bars(identity_col, identity, *, available="2026-11-14T00:00:00Z"):
    dates = pd.date_range("2026-10-01", periods=26, freq="B")
    return pd.DataFrame({
        "trade_date": dates,
        identity_col: identity,
        "close": [100.0 + i for i in range(len(dates))],
        "available_time": pd.Timestamp(available),
        "source_id": "swsresearch.index_publish.trend",
        "source_revision": [f"rev-{i}" for i in range(len(dates))],
        "availability_basis": "retrieval_time_only",
        "historical_pit_eligible": False,
    })


def test_industry_returns_are_forward_visible_only_and_keep_retrieval_lineage():
    bars = _bars("industry_code", "801010")
    before_retrieval = build_industry_index_returns(
        bars, decision_time="2026-11-13T23:59:59Z"
    )
    assert before_retrieval.empty

    after_retrieval = build_industry_index_returns(
        bars, decision_time="2026-11-14T00:00:01Z"
    )
    assert len(after_retrieval) == 6
    assert after_retrieval["industry_code"].eq("801010").all()
    assert after_retrieval["return_5"].notna().all()
    assert after_retrieval["return_20"].notna().all()
    assert (after_retrieval["available_time"] == pd.Timestamp("2026-11-14T00:00:00Z")).all()
    assert after_retrieval["historical_pit_eligible"].eq(False).all()
    assert after_retrieval["source_revision"].str.len().eq(64).all()


def test_market_adapter_requires_single_benchmark_and_maps_return_names():
    bars = _bars("index_symbol", "000001")
    result = build_market_index_returns(
        bars, decision_time="2026-11-15T00:00:00Z"
    )
    assert {"market_return_5", "market_return_20"} <= set(result.columns)
    assert result["index_symbol"].eq("000001").all()

    multiple = pd.concat([bars, _bars("index_symbol", "000300")], ignore_index=True)
    with pytest.raises(ValueError, match="exactly one"):
        build_market_index_returns(multiple, decision_time="2026-11-15T00:00:00Z")


def test_rejects_false_pit_claim_and_non_retrieval_availability():
    bars = _bars("industry_code", "801010")
    bars.loc[0, "historical_pit_eligible"] = True
    with pytest.raises(ValueError, match="PIT-ineligible"):
        build_industry_index_returns(bars, decision_time="2026-11-15T00:00:00Z")

    bars = _bars("industry_code", "801010")
    bars.loc[0, "availability_basis"] = "publication_time"
    with pytest.raises(ValueError, match="retrieval_time_only"):
        build_industry_index_returns(bars, decision_time="2026-11-15T00:00:00Z")



def test_stock_adapter_emits_daily_and_rolling_returns_for_m4_residual_model():
    bars = _bars("symbol", "000333")
    result = build_stock_daily_returns(
        bars, decision_time="2026-11-15T00:00:00Z"
    )
    assert len(result) == 6
    assert result["symbol"].eq("000333").all()
    assert {"daily_return", "return_5", "return_20", "available_time",
            "source_id", "source_revision", "historical_pit_eligible"} <= set(result.columns)
    assert result["daily_return"].notna().all()
    assert result["historical_pit_eligible"].eq(False).all()
