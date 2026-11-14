import pandas as pd
import pytest

from src.model.m4_forward_input_builder import build_forward_m4_inputs


CUTOFF = "2026-11-14T12:00:00Z"
AVAILABLE = "2026-11-14T10:00:00Z"


def _bars(identity_col, identity, periods=70):
    dates = pd.date_range("2026-08-03", periods=periods, freq="B")
    return pd.DataFrame({
        "trade_date": dates,
        identity_col: identity,
        "close": [100.0 + i * 0.7 for i in range(periods)],
        "available_time": pd.Timestamp(AVAILABLE),
        "source_id": "fixture-source",
        "source_revision": [f"r{i}" for i in range(periods)],
        "availability_basis": "retrieval_time_only",
        "historical_pit_eligible": False,
    })


def _membership(symbols=("000001", "000002"), available=AVAILABLE):
    return pd.DataFrame([{
        "snapshot_id": "snap-1",
        "expected_industry_count": 1,
        "symbol": symbol,
        "industry_code": "801010",
        "observed_at": available,
        "available_time": available,
        "source_id": "sw2021-current",
        "source_revision": "membership-r1",
        "snapshot_scope": "current_membership_observation_only",
        "availability_basis": "retrieval_time_only",
        "historical_pit_eligible": False,
    } for symbol in symbols])


def _inputs():
    stocks = pd.concat([
        _bars("symbol", "000001"),
        _bars("symbol", "000002"),
    ], ignore_index=True)
    market = _bars("index_symbol", "000001")
    industry = _bars("industry_code", "801010")
    return stocks, market, industry


def test_builds_date_aligned_forward_only_m4_frames():
    stocks, market, industry = _inputs()
    result = build_forward_m4_inputs(
        stocks, market, industry, _membership(),
        expected_symbols=["000001", "000002"],
        benchmark_symbol="000001",
        decision_time=CUTOFF,
    )
    assert result["status"] == "ready_forward_only_diagnostic"
    assert result["stock_coverage"] == 1.0
    assert result["historical_pit_eligible"] is False
    assert {"symbol", "industry_code", "daily_return", "return_5"} <= set(result["stock_returns"])
    assert {"market_return", "market_return_5"} <= set(result["market_returns"])
    assert {"industry_return", "industry_return_5"} <= set(result["industry_returns"])
    assert result["stock_returns"]["available_time"].min() >= pd.Timestamp(AVAILABLE)


def test_blocks_missing_membership_and_unknown_industry_history():
    stocks, market, industry = _inputs()
    with pytest.raises(ValueError, match="incomplete_universe"):
        build_forward_m4_inputs(
            stocks, market, industry, _membership(("000001",)),
            expected_symbols=["000001", "000002"],
            benchmark_symbol="000001", decision_time=CUTOFF,
        )

    membership = _membership()
    membership["industry_code"] = "809999"
    with pytest.raises(ValueError, match="industry index history missing"):
        build_forward_m4_inputs(
            stocks, market, industry, membership,
            expected_symbols=["000001", "000002"],
            benchmark_symbol="000001", decision_time=CUTOFF,
        )


def test_blocks_stale_snapshot_and_missing_benchmark():
    stocks, market, industry = _inputs()
    stale = _membership(available="2026-11-10T10:00:00Z")
    with pytest.raises(ValueError, match="stale_snapshot"):
        build_forward_m4_inputs(
            stocks, market, industry, stale,
            expected_symbols=["000001", "000002"],
            benchmark_symbol="000001", decision_time=CUTOFF,
        )

    with pytest.raises(ValueError, match="benchmark is absent"):
        build_forward_m4_inputs(
            stocks, market, industry, _membership(),
            expected_symbols=["000001", "000002"],
            benchmark_symbol="000300", decision_time=CUTOFF,
        )


def test_blocks_ambiguous_multiple_industry_assignments_and_short_stock_history():
    stocks, market, industry = _inputs()
    multi = pd.concat([_membership(), _membership().assign(industry_code="801020")],
                      ignore_index=True)
    multi["expected_industry_count"] = 2
    with pytest.raises(ValueError, match="multiple current industry"):
        build_forward_m4_inputs(
            stocks, market, industry, multi,
            expected_symbols=["000001", "000002"],
            benchmark_symbol="000001", decision_time=CUTOFF,
        )

    short_stocks = stocks.loc[stocks["symbol"] == "000001"].copy()
    short_stocks = pd.concat([short_stocks, stocks.loc[stocks["symbol"] == "000002"].iloc[:20]])
    with pytest.raises(ValueError, match="insufficient stock history"):
        build_forward_m4_inputs(
            short_stocks, market, industry, _membership(),
            expected_symbols=["000001", "000002"],
            benchmark_symbol="000001", decision_time=CUTOFF,
        )
