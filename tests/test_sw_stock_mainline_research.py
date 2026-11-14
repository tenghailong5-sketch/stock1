import pandas as pd
import pytest

from src.model.sw_stock_mainline_research import build_sw_stock_mainline_research


def _membership():
    return pd.DataFrame([
        {
            "symbol": "600000", "industry_code": "801010",
            "valid_from": "2020-01-01", "valid_to": None,
            "availability_basis": "retrieval_time_only",
            "historical_pit_eligible": False,
        },
        {
            "symbol": "000001", "industry_code": "801020",
            "valid_from": "2020-01-01", "valid_to": None,
            "availability_basis": "retrieval_time_only",
            "historical_pit_eligible": False,
        },
    ])


def _mainline():
    return pd.DataFrame([
        {
            "industry_code": "801010", "trade_date": "2026-11-13",
            "mainline_phase": "IGNITION", "strength_percentile": 0.95,
            "turnover_share": 0.08,
        },
        {
            "industry_code": "801020", "trade_date": "2026-11-13",
            "mainline_phase": "DECAY", "strength_percentile": 0.2,
            "turnover_share": 0.01,
        },
    ])


def test_maps_effective_stock_classification_to_latest_mainline_research():
    result, audit = build_sw_stock_mainline_research(
        _membership(), _mainline(), as_of_date="2026-11-15"
    )
    assert result.set_index("symbol").loc["600000", "mainline_phase"] == "IGNITION"
    assert result.set_index("symbol").loc["000001", "mainline_phase"] == "DECAY"
    assert audit["status"] == "READY_RESEARCH_ONLY"
    assert audit["membership_to_mainline_coverage"] == 1.0
    assert not result["historical_pit_eligible"].any()
    assert not result["trading_authorization"].any()


def test_excludes_membership_outside_effective_interval():
    members = _membership()
    members.loc[members.symbol.eq("600000"), "valid_to"] = "2026-01-01"
    result, audit = build_sw_stock_mainline_research(
        members, _mainline(), as_of_date="2026-11-15"
    )
    assert "600000" not in set(result.symbol)
    assert audit["active_membership_symbols"] == 1


def test_duplicate_active_membership_fails_closed():
    members = pd.concat([_membership(), _membership().iloc[[0]]], ignore_index=True)
    result, audit = build_sw_stock_mainline_research(
        members, _mainline(), as_of_date="2026-11-15"
    )
    assert result.empty
    assert audit["status"] == "AMBIGUOUS_STOCK_INDUSTRY_MAPPING"
    assert audit["trading_authorization"] is False


def test_missing_mainline_industry_is_reported_as_incomplete_coverage():
    panel = _mainline().iloc[[0]].copy()
    result, audit = build_sw_stock_mainline_research(
        _membership(), panel, as_of_date="2026-11-15"
    )
    assert audit["status"] == "INCOMPLETE_RESEARCH_COVERAGE"
    assert audit["matched_symbols"] == 1
    assert audit["membership_to_mainline_coverage"] == 0.5


def test_rejects_pit_eligible_membership_rows():
    members = _membership()
    members.loc[0, "historical_pit_eligible"] = True
    with pytest.raises(ValueError, match="PIT-eligible"):
        build_sw_stock_mainline_research(
            members, _mainline(), as_of_date="2026-11-15"
        )
