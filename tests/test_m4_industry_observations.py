import pandas as pd
import pytest

from src.model.m4_industry_observations import select_latest_observed_industry_snapshot


def rows(snapshot_id="batch-a", captured="2026-11-12T08:30:00Z"):
    common = {
        "snapshot_id": snapshot_id,
        "expected_industry_count": 2,
        "observed_at": captured,
        "available_time": captured,
        "source_id": "akshare.stock_board_industry_cons_em",
        "source_revision": "rev",
        "snapshot_scope": "current_membership_observation_only",
        "availability_basis": "retrieval_time_only",
        "historical_pit_eligible": False,
    }
    return pd.DataFrame([
        {**common, "symbol": "000001", "industry_code": "BK01"},
        {**common, "symbol": "000001", "industry_code": "BK02"},
        {**common, "symbol": "600000", "industry_code": "BK01"},
    ])


def test_selects_latest_visible_batch_and_preserves_multiple_industries():
    old = rows("old", "2026-11-11T08:30:00Z")
    new = rows("new", "2026-11-12T08:30:00Z")
    selected, audit = select_latest_observed_industry_snapshot(
        pd.concat([old, new], ignore_index=True),
        ["000001", "600000"],
        decision_time="2026-11-12T09:00:00Z",
        max_age="1D",
    )
    assert audit["status"] == "ready_forward_only"
    assert audit["snapshot_id"] == "new"
    assert len(selected) == 3
    assert set(selected.loc[selected.symbol == "000001", "industry_code"]) == {"BK01", "BK02"}
    assert selected.historical_pit_eligible.eq(False).all()


def test_future_batch_is_not_visible():
    selected, audit = select_latest_observed_industry_snapshot(
        rows(captured="2026-11-12T09:01:00Z"),
        ["000001", "600000"],
        decision_time="2026-11-12T09:00:00Z",
    )
    assert selected.empty
    assert audit["status"] == "no_visible_snapshot"


def test_latest_visible_incomplete_batch_fails_without_old_batch_fallback():
    old = rows("old", "2026-11-11T08:30:00Z")
    new = rows("new", "2026-11-12T08:30:00Z").iloc[:2]
    selected, audit = select_latest_observed_industry_snapshot(
        pd.concat([old, new], ignore_index=True),
        ["000001", "600000"],
        decision_time="2026-11-12T09:00:00Z",
    )
    assert selected.empty
    assert audit["status"] == "incomplete_universe"


def test_duplicate_symbol_industry_in_latest_batch_fails_closed():
    frame = rows()
    frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    selected, audit = select_latest_observed_industry_snapshot(
        frame, ["000001", "600000"], decision_time="2026-11-12T09:00:00Z"
    )
    assert selected.empty
    assert audit["status"] == "duplicate_membership"


def test_stale_latest_batch_fails_closed():
    selected, audit = select_latest_observed_industry_snapshot(
        rows(captured="2026-11-09T08:30:00Z"),
        ["000001", "600000"],
        decision_time="2026-11-12T09:00:00Z",
        max_age="1D",
    )
    assert selected.empty
    assert audit["status"] == "stale_snapshot"


def test_naive_timestamps_are_rejected():
    frame = rows()
    frame["available_time"] = "2026-11-12 08:30:00"
    with pytest.raises(ValueError, match="timezone-aware"):
        select_latest_observed_industry_snapshot(
            frame, ["000001", "600000"], decision_time="2026-11-12T09:00:00Z"
        )


def test_historical_pit_rows_are_rejected():
    frame = rows()
    frame.loc[0, "historical_pit_eligible"] = True
    with pytest.raises(ValueError, match="historical PIT"):
        select_latest_observed_industry_snapshot(
            frame, ["000001", "600000"], decision_time="2026-11-12T09:00:00Z"
        )



def test_batch_missing_an_expected_industry_is_rejected_even_if_universe_is_covered():
    frame = rows()
    frame = frame.loc[frame["industry_code"] == "BK01"].copy()
    selected, audit = select_latest_observed_industry_snapshot(
        frame, ["000001", "600000"], decision_time="2026-11-12T09:00:00Z"
    )
    assert selected.empty
    assert audit["status"] == "incomplete_industry_batch"


def test_inconsistent_batch_manifest_is_rejected():
    frame = rows()
    frame.loc[0, "expected_industry_count"] = 3
    selected, audit = select_latest_observed_industry_snapshot(
        frame, ["000001", "600000"], decision_time="2026-11-12T09:00:00Z"
    )
    assert selected.empty
    assert audit["status"] == "invalid_batch_manifest"
