import pandas as pd
import pytest

from src.fhm100 import (
    select_history_universe,
    select_history_universe_with_supplement,
)


def test_history_pool_is_liquidity_ranked_and_bounded_without_losing_market_snapshot():
    snapshot = pd.DataFrame([
        {"code": "000001", "name": "A", "price": 10, "amount": 100, "pct_chg": 1},
        {"code": "688001", "name": "B", "price": 20, "amount": 500, "pct_chg": 2},
        {"code": "830001", "name": "C", "price": 30, "amount": 300, "pct_chg": 3},
        {"code": "430001", "name": "D", "price": 40, "amount": 200, "pct_chg": 4},
    ])

    selected = select_history_universe(snapshot, max_symbols=2)

    assert selected["code"].tolist() == ["688001", "830001"]
    assert len(snapshot) == 4  # the caller retains the complete live quote snapshot


def test_history_pool_normalizes_market_prefix_and_drops_invalid_quotes():
    snapshot = pd.DataFrame([
        {"code": "sh600000", "name": "A", "price": 10, "amount": 100, "pct_chg": 1},
        {"code": "sz000001", "name": "B", "price": 0, "amount": 500, "pct_chg": 2},
        {"code": "bad", "name": "C", "price": 30, "amount": 300, "pct_chg": 3},
    ])

    selected = select_history_universe(snapshot, max_symbols=10)

    assert selected["code"].tolist() == ["600000"]


def test_history_pool_rejects_nonpositive_limit():
    snapshot = pd.DataFrame(columns=["code", "name", "price", "amount", "pct_chg"])
    with pytest.raises(ValueError, match="max_symbols"):
        select_history_universe(snapshot, max_symbols=0)



def test_quote_prefilter_adds_qualified_symbols_outside_primary_pool_and_audits_deferred():
    snapshot = pd.DataFrame([
        {"code": "000001", "name": "A", "price": 10, "amount": 900, "pct_chg": -1},
        {"code": "000002", "name": "B", "price": 10, "amount": 800, "pct_chg": 0.2},
        {"code": "000003", "name": "C", "price": 10, "amount": 700, "pct_chg": 1.0},
        {"code": "000004", "name": "D", "price": 10, "amount": 600, "pct_chg": 2.0},
        {"code": "000005", "name": "E", "price": 10, "amount": 50, "pct_chg": 3.0},
    ])

    selected, audit = select_history_universe_with_supplement(
        snapshot, max_symbols=1, max_supplement_symbols=1, min_candidate_amount=100,
    )

    assert selected["code"].tolist() == ["000001", "000002"]
    assert audit["primary_history_symbols"] == 1
    assert audit["supplement_eligible_symbols"] == 3
    assert audit["supplement_selected_symbols"] == 1
    assert audit["supplement_deferred_symbols"] == 2
    assert audit["supplement_is_formal_candidate"] is False


def test_quote_prefilter_does_not_duplicate_primary_or_admit_nonpositive_change():
    snapshot = pd.DataFrame([
        {"code": "000001", "name": "A", "price": 10, "amount": 900, "pct_chg": 0},
        {"code": "000002", "name": "B", "price": 10, "amount": 800, "pct_chg": 1},
        {"code": "000003", "name": "C", "price": 10, "amount": 700, "pct_chg": -1},
    ])

    selected, audit = select_history_universe_with_supplement(
        snapshot, max_symbols=1, max_supplement_symbols=10, min_candidate_amount=100,
    )

    assert selected["code"].tolist() == ["000001", "000002"]
    assert selected["code"].is_unique
    assert audit["supplement_eligible_symbols"] == 1


def test_quote_prefilter_rejects_negative_supplement_limit():
    snapshot = pd.DataFrame(columns=["code", "name", "price", "amount", "pct_chg"])
    with pytest.raises(ValueError, match="max_supplement_symbols"):
        select_history_universe_with_supplement(snapshot, max_supplement_symbols=-1)
