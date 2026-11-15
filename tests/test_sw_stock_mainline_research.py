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


def test_forward_current_membership_maps_stock_to_mainline_without_pit_authority():
    from src.model.sw_stock_mainline_research import build_forward_stock_mainline_research

    membership = pd.DataFrame([
        {"symbol": "600000", "industry_level1": "银行", "observed_at": "2026-11-14T13:00:00Z",
         "availability_basis": "retrieval_time_only", "historical_pit_eligible": False,
         "source_id": "swsresearch.sw2021_stock_classification", "source_revision": "rev-a",
         "snapshot_scope": "current_membership_observation_only", "snapshot_id": "snap-1"},
        {"symbol": "000001", "industry_level1": "房地产", "observed_at": "2026-11-14T13:00:00Z",
         "availability_basis": "retrieval_time_only", "historical_pit_eligible": False,
         "source_id": "swsresearch.sw2021_stock_classification", "source_revision": "rev-b",
         "snapshot_scope": "current_membership_observation_only", "snapshot_id": "snap-1"},
    ])
    mainline = pd.DataFrame([
        {"industry_code": "801780", "industry_name": "银行", "trade_date": "2026-11-13",
         "mainline_phase": "IGNITION", "mainline_phase_gated": "IGNITION_UNCONFIRMED",
         "strength_percentile": 0.9},
        {"industry_code": "801180", "industry_name": "房地产", "trade_date": "2026-11-13",
         "mainline_phase": "DECAY", "mainline_phase_gated": "DECAY",
         "strength_percentile": 0.2},
    ])
    result, audit = build_forward_stock_mainline_research(
        membership, mainline, decision_time="2026-11-15T10:00:00Z"
    )
    assert result.set_index("symbol").loc["600000", "mainline_phase"] == "IGNITION"
    assert result.set_index("symbol").loc["000001", "mainline_phase"] == "DECAY"
    assert audit["status"] == "READY_RESEARCH_ONLY"
    assert audit["membership_to_mainline_coverage"] == 1.0
    assert not result["historical_pit_eligible"].any()
    assert not result["trading_authorization"].any()


def test_forward_mapper_preserves_membership_industry_code_without_collision():
    from src.model.sw_stock_mainline_research import build_forward_stock_mainline_research

    membership = pd.DataFrame([{
        "symbol": "600000", "industry_code": "SWL1:银行",
        "industry_name": "银行", "industry_level1": "银行",
        "observed_at": "2026-11-14T13:00:00Z",
        "availability_basis": "retrieval_time_only", "historical_pit_eligible": False,
        "source_id": "snapshot", "source_revision": "rev-1",
        "snapshot_scope": "current_membership_observation_only",
    }])
    mainline = pd.DataFrame([{
        "industry_code": "SWL1:银行", "industry_name": "银行",
        "trade_date": "2026-11-13", "mainline_phase": "UNKNOWN",
        "mainline_phase_gated": "UNKNOWN", "strength_percentile": 0.8,
    }])
    result, audit = build_forward_stock_mainline_research(
        membership, mainline, decision_time="2026-11-15T10:00:00Z"
    )
    assert result.loc[0, "industry_code"] == "SWL1:银行"
    assert result.loc[0, "membership_industry_code"] == "SWL1:银行"
    assert audit["mapped_industry_count"] == 1


def test_forward_mapper_rejects_future_membership_snapshot():
    from src.model.sw_stock_mainline_research import build_forward_stock_mainline_research

    membership = pd.DataFrame([{
        "symbol": "600000", "industry_level1": "银行", "observed_at": "2026-11-16T13:00:00Z",
        "availability_basis": "retrieval_time_only", "historical_pit_eligible": False,
        "source_id": "swsresearch.sw2021_stock_classification", "source_revision": "rev-a",
        "snapshot_scope": "current_membership_observation_only", "snapshot_id": "snap-1",
    }])
    mainline = pd.DataFrame([{
        "industry_code": "801780", "industry_name": "银行", "trade_date": "2026-11-13",
        "mainline_phase": "IGNITION", "mainline_phase_gated": "IGNITION_UNCONFIRMED",
        "strength_percentile": 0.9,
    }])
    result, audit = build_forward_stock_mainline_research(
        membership, mainline, decision_time="2026-11-15T10:00:00Z"
    )
    assert result.empty
    assert audit["status"] == "NO_CURRENT_MAPPING_INPUTS"



def test_spot_industry_strength_fallback_ranks_sectors_without_inventing_phase():
    from src.model.sw_stock_mainline_research import build_spot_industry_strength_panel

    observed = "2026-11-13T08:00:00Z"
    membership = pd.DataFrame({
        "symbol": ["000001", "000002", "000003", "000004"],
        "industry_level1": ["银行", "银行", "电子", "电子"],
        "observed_at": [observed] * 4,
        "availability_basis": ["retrieval_time_only"] * 4,
        "historical_pit_eligible": [False] * 4,
        "snapshot_scope": ["current_membership_observation_only"] * 4,
    })
    stocks = pd.DataFrame({
        "code": ["000001", "000002", "000003", "000004"],
        "pct_chg": [0.5, 1.0, 8.0, 6.0],
        "amount": [100.0, 100.0, 100.0, 100.0],
    })
    panel, audit = build_spot_industry_strength_panel(
        membership, stocks,
        trade_date="2026-11-13",
        decision_time="2026-11-13T09:00:00Z",
        snapshot_observed_at="2026-11-13T08:30:00Z",
    )
    assert audit["status"] == "READY_SPOT_ONLY_RESEARCH"
    assert panel.iloc[0].industry_name == "电子"
    assert panel.iloc[0].strength_percentile == 1.0
    assert panel.mainline_phase.eq("UNKNOWN").all()
    assert panel.mainline_phase_available.eq(False).all()
    assert panel.historical_pit_eligible.eq(False).all()
    assert panel.trading_authorization.eq(False).all()


def test_spot_industry_strength_fallback_rejects_future_quote_snapshot():
    from src.model.sw_stock_mainline_research import build_spot_industry_strength_panel

    membership = pd.DataFrame({
        "symbol": ["000001"], "industry_level1": ["银行"],
        "observed_at": ["2026-11-13T08:00:00Z"],
        "availability_basis": ["retrieval_time_only"],
        "historical_pit_eligible": [False],
        "snapshot_scope": ["current_membership_observation_only"],
    })
    stocks = pd.DataFrame({"code": ["000001"], "pct_chg": [1.0], "amount": [100.0]})
    try:
        build_spot_industry_strength_panel(
            membership, stocks, trade_date="2026-11-13",
            decision_time="2026-11-13T08:00:00Z",
            snapshot_observed_at="2026-11-13T08:30:00Z",
        )
    except ValueError as exc:
        assert "newer than decision_time" in str(exc)
    else:
        raise AssertionError("future stock quote snapshots must be rejected")



def test_forward_mapper_carries_live_m3_persistence_readiness_to_each_stock():
    from src.model.sw_stock_mainline_research import build_forward_stock_mainline_research

    membership = pd.DataFrame([
        {"symbol": "600000", "industry_level1": "银行", "observed_at": "2026-11-14T13:00:00Z",
         "availability_basis": "retrieval_time_only", "historical_pit_eligible": False,
         "source_id": "snapshot", "source_revision": "rev-a",
         "snapshot_scope": "current_membership_observation_only"},
        {"symbol": "000001", "industry_level1": "房地产", "observed_at": "2026-11-14T13:00:00Z",
         "availability_basis": "retrieval_time_only", "historical_pit_eligible": False,
         "source_id": "snapshot", "source_revision": "rev-b",
         "snapshot_scope": "current_membership_observation_only"},
    ])
    mainline = pd.DataFrame([
        {"industry_code": "801780", "industry_name": "银行", "trade_date": "2026-11-13",
         "mainline_phase": "IGNITION", "mainline_phase_gated": "IGNITION",
         "strength_percentile": 0.9, "persistence_ready": True},
        {"industry_code": "801180", "industry_name": "房地产", "trade_date": "2026-11-13",
         "mainline_phase": "IGNITION", "mainline_phase_gated": "IGNITION",
         "strength_percentile": 0.8, "persistence_ready": False},
    ])
    result, _ = build_forward_stock_mainline_research(
        membership, mainline, decision_time="2026-11-15T10:00:00Z"
    )
    readiness = result.set_index("symbol")["persistence_ready"].to_dict()
    assert readiness == {"000001": False, "600000": True}
