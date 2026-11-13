import pandas as pd
import pytest

from src.fhm100 import select_history_universe


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
