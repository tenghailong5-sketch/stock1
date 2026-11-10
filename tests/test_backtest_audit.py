import pandas as pd
import pytest

from src.model.backtest_audit import audit_backtest_dataset, require_backtest_ready


def datasets():
    decision = pd.Timestamp("2026-01-05T15:00:00+08:00")
    candidates = pd.DataFrame([{
        "symbol": "000001", "decision_time": decision,
        "available_time": pd.Timestamp("2026-01-05T14:59:00+08:00"),
        "historical_pit_eligible": True, "source_revision": "candidate-r1",
        "strategy": "A", "score": 88.0, "eligible": True,
        "corporate_actions_resolved": True, "universe_complete": True,
        "delisted_included": True, "ruleset_version": "FHM-100-test",
    }])
    bars = pd.DataFrame([{
        "symbol": "000001", "trade_date": "2026-01-06",
        "available_time": pd.Timestamp("2026-01-06T15:01:00+08:00"),
        "historical_pit_eligible": True, "source_revision": "bar-r1",
        "open": 10.0, "high": 10.5, "low": 9.8, "close": 10.2,
        "volume": 100000, "suspended": False, "limit_up_locked": False,
        "limit_down_locked": False, "corporate_actions_resolved": True,
        "universe_complete": True, "delisted_included": True,
    }])
    return candidates, bars


def test_complete_pit_dataset_passes_admission_audit():
    candidates, bars = datasets()
    audit = audit_backtest_dataset(candidates, bars)
    assert audit.ready
    assert audit.errors == ()
    assert require_backtest_ready(candidates, bars).ready


def test_retrieval_time_only_data_is_blocked():
    candidates, bars = datasets()
    bars.loc[0, "historical_pit_eligible"] = False
    audit = audit_backtest_dataset(candidates, bars)
    assert not audit.ready
    assert "bar_contains_non_pit_observations" in audit.errors
    with pytest.raises(ValueError, match="historical backtest blocked"):
        require_backtest_ready(candidates, bars)


def test_candidate_available_after_decision_is_lookahead():
    candidates, bars = datasets()
    candidates.loc[0, "available_time"] = pd.Timestamp("2026-01-05T15:01:00+08:00")
    audit = audit_backtest_dataset(candidates, bars)
    assert not audit.ready
    assert "candidate_lookahead_available_after_decision" in audit.errors


def test_missing_delisted_or_corporate_action_certification_blocks():
    candidates, bars = datasets()
    bars.loc[0, "delisted_included"] = False
    candidates.loc[0, "corporate_actions_resolved"] = False
    audit = audit_backtest_dataset(candidates, bars)
    assert not audit.ready
    assert "bar_not_certified:delisted_included" in audit.errors
    assert "candidate_not_certified:corporate_actions_resolved" in audit.errors


def test_invalid_ohlcv_and_low_score_candidate_block():
    candidates, bars = datasets()
    bars.loc[0, "high"] = 9.0
    candidates.loc[0, "score"] = 79.99
    audit = audit_backtest_dataset(candidates, bars)
    assert not audit.ready
    assert "bar_invalid_high" in audit.errors
    assert "eligible_candidate_score_outside_80_100" in audit.errors


def test_duplicate_symbol_decision_strategy_is_rejected():
    candidates, bars = datasets()
    candidates = pd.concat([candidates, candidates], ignore_index=True)
    audit = audit_backtest_dataset(candidates, bars)
    assert not audit.ready
    assert "candidate_duplicate_natural_key" in audit.errors

def test_execution_flags_must_be_explicit_booleans():
    candidates, bars = datasets()
    bars["limit_down_locked"] = bars["limit_down_locked"].astype(object)
    bars.loc[0, "limit_down_locked"] = None
    audit = audit_backtest_dataset(candidates, bars)
    assert not audit.ready
    assert "bar_invalid_boolean:limit_down_locked" in audit.errors


def test_unknown_strategy_is_rejected():
    candidates, bars = datasets()
    candidates.loc[0, "strategy"] = "D"
    audit = audit_backtest_dataset(candidates, bars)
    assert not audit.ready
    assert "candidate_contains_unknown_strategy" in audit.errors
