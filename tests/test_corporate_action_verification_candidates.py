import pandas as pd

from src.model.corporate_action_verification_candidates import (
    build_corporate_action_verification_candidates,
)


def _ledger():
    return pd.DataFrame([{
        "symbol": "600000", "event_type": "dividend_bonus",
        "event_key": "event-1", "announcement_date": "2026-01-01",
        "ex_date": "2026-02-01", "cash_dividend": 2.0,
        "bonus_share_ratio": 1.0, "capitalization_ratio": 2.0,
        "rights_issue_ratio": None, "rights_issue_price": None,
        "first_seen_at": "2026-11-15T00:00:00Z",
        "available_time": "2026-11-15T00:00:00Z",
        "source_id": "akshare.stock_history_dividend_detail",
        "source_revision": "rev-1", "corporate_action_status": "CAPTURED",
        "progress": "实施",
    }])


def test_dividend_components_normalize_from_per_ten_to_per_share_but_stay_pending():
    out = build_corporate_action_verification_candidates(
        _ledger(), decision_time="2026-11-16T00:00:00Z",
        ledger_snapshot_id="snapshot-1",
    )
    row = out.iloc[0]
    assert row.cash_per_share == 0.2
    assert row.share_factor == 1.3
    assert row.verified_status == "PENDING_VERIFICATION"
    assert bool(row.verification_required)
    assert row.ledger_snapshot_id == "snapshot-1"


def test_event_is_not_available_before_first_observation():
    out = build_corporate_action_verification_candidates(
        _ledger(), decision_time="2026-11-14T23:59:59Z",
        ledger_snapshot_id="snapshot-1",
    )
    assert out.empty


def test_rights_issue_is_never_simplified_to_free_share_factor():
    ledger = _ledger()
    ledger.loc[0, "event_type"] = "rights_issue"
    ledger.loc[0, "rights_issue_ratio"] = 3.0
    ledger.loc[0, "rights_issue_price"] = 5.0
    out = build_corporate_action_verification_candidates(
        ledger, decision_time="2026-11-16T00:00:00Z",
        ledger_snapshot_id="snapshot-1",
    )
    row = out.iloc[0]
    assert row.action_type == "unsupported"
    assert row.share_factor is None
    assert row.verified_status == "PENDING_VERIFICATION"
    assert "subscription_and_takeup" in row.candidate_reason


def test_unimplemented_plan_is_not_a_verification_candidate():
    ledger = _ledger()
    ledger.loc[0, "progress"] = "预案"
    out = build_corporate_action_verification_candidates(
        ledger, decision_time="2026-11-16T00:00:00Z",
        ledger_snapshot_id="snapshot-1",
    )
    assert out.iloc[0].candidate_reason == "event_not_confirmed_implemented"
    assert out.iloc[0].verified_status == "PENDING_VERIFICATION"


def test_naive_decision_time_fails_closed():
    import pytest
    with pytest.raises(ValueError, match="timezone-aware"):
        build_corporate_action_verification_candidates(
            _ledger(), decision_time="2026-11-16 00:00:00",
            ledger_snapshot_id="snapshot-1",
        )
