import pandas as pd
import pytest

from src.model.pit_corporate_actions import build_pit_total_return_prices


CUTOFF = "2026-11-12T01:00:00Z"


def fixtures():
    prices = pd.DataFrame({
        "symbol": ["600000.SH"] * 3,
        "trade_date": ["2026-11-09", "2026-11-10", "2026-11-11"],
        "open": [10.0, 9.4, 9.6],
        "high": [10.2, 10.0, 9.9],
        "low": [9.8, 9.2, 9.4],
        "close": [10.0, 9.5, 9.8],
        "available_time": ["2026-11-09T08:00:00Z", "2026-11-10T08:00:00Z", "2026-11-11T08:00:00Z"],
        "source_id": ["test"] * 3,
        "source_revision": ["p1", "p2", "p3"],
    })
    actions = pd.DataFrame({
        "symbol": ["600000.SH"],
        "ex_date": ["2026-11-10"],
        "action_type": ["cash_dividend"],
        "cash_per_share": [1.0],
        "share_factor": [1.0],
        "verified_status": ["VERIFIED"],
        "available_time": ["2026-11-10T09:00:00Z"],
        "source_id": ["test-actions"],
        "source_revision": ["a1"],
        "ledger_snapshot_id": ["ledger-20261111-01"],
    })
    coverage = pd.DataFrame({
        "symbol": ["600000.SH"],
        "coverage_start": ["2026-11-09"],
        "coverage_end": ["2026-11-11"],
        "coverage_status": ["COMPLETE"],
        "available_time": ["2026-11-11T09:00:00Z"],
        "source_id": ["test-coverage"],
        "source_revision": ["c1"],
        "ledger_snapshot_id": ["ledger-20261111-01"],
    })
    return prices, actions, coverage


def test_builds_total_return_index_from_verified_action_and_complete_pit_coverage():
    prices, actions, coverage = fixtures()
    out = build_pit_total_return_prices(prices, actions, coverage, decision_time=CUTOFF)
    assert out.corporate_action_status.eq("VERIFIED").all()
    assert out.pit_total_return_close.iloc[0] == 100.0
    # A 1.00 cash dividend offsets the raw close drop from 10.00 to 9.50.
    assert out.pit_total_return_close.iloc[1] == 105.0
    assert out.pit_total_return_close.iloc[2] == pytest.approx(108.3157894736842)
    assert out.pit_total_return_high.iloc[1] == 110.0
    assert out.pit_total_return_available_time.max() <= pd.Timestamp(CUTOFF)


def test_missing_or_unknown_coverage_fails_closed_even_when_no_actions_returned():
    prices, _, coverage = fixtures()
    coverage.loc[0, "coverage_status"] = "unknown"
    out = build_pit_total_return_prices(
        prices, pd.DataFrame(columns=[
            "symbol", "ex_date", "action_type", "cash_per_share", "share_factor",
            "verified_status", "available_time", "source_id", "source_revision", "ledger_snapshot_id"
        ]), coverage, decision_time=CUTOFF,
    )
    assert out.corporate_action_status.eq("COVERAGE_NOT_VERIFIED").all()
    assert out.pit_total_return_close.isna().all()


def test_relevant_action_retrieved_after_cutoff_blocks_series_instead_of_being_ignored():
    prices, actions, coverage = fixtures()
    actions.loc[0, "available_time"] = "2026-11-12T02:00:00Z"
    out = build_pit_total_return_prices(prices, actions, coverage, decision_time=CUTOFF)
    assert out.corporate_action_status.eq("ACTION_NOT_AVAILABLE_AT_DECISION").all()
    assert out.pit_total_return_close.isna().all()


def test_coverage_attestation_after_cutoff_blocks_series():
    prices, actions, coverage = fixtures()
    coverage.loc[0, "available_time"] = "2026-11-12T02:00:00Z"
    out = build_pit_total_return_prices(prices, actions, coverage, decision_time=CUTOFF)
    assert out.corporate_action_status.eq("COVERAGE_NOT_VERIFIED").all()
    assert out.pit_total_return_close.isna().all()


def test_action_on_first_price_date_blocks_missing_prior_close():
    prices, actions, coverage = fixtures()
    actions.loc[0, "ex_date"] = "2026-11-09"
    out = build_pit_total_return_prices(prices, actions, coverage, decision_time=CUTOFF)
    assert out.corporate_action_status.eq("ACTION_BEFORE_PRICE_COVERAGE").all()
    assert out.pit_total_return_close.isna().all()


def test_naive_price_availability_is_rejected():
    prices, actions, coverage = fixtures()
    prices.loc[0, "available_time"] = "2026-11-09 08:00:00"
    out = build_pit_total_return_prices(prices, actions, coverage, decision_time=CUTOFF)
    assert out.corporate_action_status.eq("PIT_BUILD_REJECTED").all()
    assert out.pit_total_return_close.isna().all()



def test_action_rows_must_match_the_attested_ledger_snapshot():
    prices, actions, coverage = fixtures()
    actions.loc[0, "ledger_snapshot_id"] = "different-ledger"
    out = build_pit_total_return_prices(prices, actions, coverage, decision_time=CUTOFF)
    assert out.corporate_action_status.eq("INVALID_ACTION_LINEAGE").all()
    assert out.pit_total_return_close.isna().all()
