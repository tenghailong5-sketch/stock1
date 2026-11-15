import pandas as pd
import pytest

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
        "observed_at": "2026-11-15T00:00:00Z",
        "available_time": "2026-11-15T00:00:00Z",
        "source_id": "akshare.stock_history_dividend_detail",
        "source_revision": "rev-1", "corporate_action_status": "CAPTURED",
        "progress": "实施",
    }])


def _build(ledger, cutoff="2026-11-16T00:00:00Z"):
    return build_corporate_action_verification_candidates(
        ledger, decision_time=cutoff, ledger_snapshot_id="snapshot-1",
    )


def test_dividend_components_normalize_per_ten_but_stay_pending():
    row = _build(_ledger()).iloc[0]
    assert row.cash_per_share == 0.2
    assert row.share_factor == 1.3
    assert row.verified_status == "PENDING_VERIFICATION"
    assert bool(row.verification_required)
    assert row.ledger_snapshot_id == "snapshot-1"
    assert row.available_time == pd.Timestamp("2026-11-15T00:00:00Z")


def test_event_is_not_available_before_first_observation():
    assert _build(_ledger(), "2026-11-14T23:59:59Z").empty


def test_revision_is_not_backdated_to_first_seen_time():
    ledger = _ledger()
    revised = ledger.copy()
    revised.loc[0, "cash_dividend"] = 3.0
    revised.loc[0, "source_revision"] = "rev-2"
    revised.loc[0, "observed_at"] = "2026-11-16T00:00:00Z"
    revised.loc[0, "available_time"] = "2026-11-15T00:00:00Z"
    both = pd.concat([ledger, revised], ignore_index=True)

    before_revision = _build(both, "2026-11-15T23:59:59Z").iloc[0]
    after_revision = _build(both, "2026-11-16T00:00:00Z").iloc[0]
    assert before_revision.source_revision == "rev-1"
    assert before_revision.cash_per_share == 0.2
    assert after_revision.source_revision == "rev-2"
    assert after_revision.cash_per_share == 0.3
    assert after_revision.available_time == pd.Timestamp("2026-11-16T00:00:00Z")


def test_latest_revision_is_selected_once_for_each_event():
    ledger = _ledger()
    revised = ledger.copy()
    revised.loc[0, "cash_dividend"] = 3.0
    revised.loc[0, "source_revision"] = "rev-2"
    revised.loc[0, "observed_at"] = "2026-11-16T00:00:00Z"
    out = _build(pd.concat([ledger, revised], ignore_index=True))
    assert len(out) == 1
    assert out.iloc[0].source_revision == "rev-2"


def test_conflicting_revisions_at_same_observation_time_fail_closed():
    ledger = _ledger()
    conflicting = ledger.copy()
    conflicting.loc[0, "cash_dividend"] = 9.0
    conflicting.loc[0, "source_revision"] = "rev-conflict"
    with pytest.raises(ValueError, match="conflicting revisions"):
        _build(pd.concat([ledger, conflicting], ignore_index=True))


def test_rights_issue_is_never_simplified_to_free_share_factor():
    ledger = _ledger()
    ledger.loc[0, "event_type"] = "rights_issue"
    ledger.loc[0, "rights_issue_ratio"] = 3.0
    ledger.loc[0, "rights_issue_price"] = 5.0
    row = _build(ledger).iloc[0]
    assert row.action_type == "unsupported"
    assert pd.isna(row.share_factor)
    assert row.verified_status == "PENDING_VERIFICATION"
    assert "subscription_and_takeup" in row.candidate_reason


def test_unimplemented_plan_is_not_a_verification_candidate():
    ledger = _ledger()
    ledger.loc[0, "progress"] = "预案"
    row = _build(ledger).iloc[0]
    assert row.candidate_reason == "event_not_confirmed_implemented"
    assert row.verified_status == "PENDING_VERIFICATION"


def test_naive_decision_time_fails_closed():
    with pytest.raises(ValueError, match="timezone-aware"):
        _build(_ledger(), "2026-11-16 00:00:00")


def test_progress_and_revision_observation_are_required():
    with pytest.raises(ValueError, match="progress"):
        _build(_ledger().drop(columns=["progress"]))
    with pytest.raises(ValueError, match="observed_at"):
        _build(_ledger().drop(columns=["observed_at"]))
