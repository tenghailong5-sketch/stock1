import pandas as pd

from src.fhm100 import select_research_history_universe


def test_research_history_pool_balances_liquidity_momentum_and_pullbacks():
    snapshot = pd.DataFrame([
        {"code": f"{i:06d}", "name": f"stock-{i}", "price": 10.0,
         "amount": 1000.0 - i, "pct_chg": 0.1}
        for i in range(1, 9)
    ] + [
        {"code": "000101", "name": "active-turnover", "price": 10.0,
         "amount": 500.0, "pct_chg": 2.0},
        {"code": "000102", "name": "active-turnover-2", "price": 10.0,
         "amount": 490.0, "pct_chg": 1.0},
        {"code": "000103", "name": "momentum-leader", "price": 10.0,
         "amount": 100.0, "pct_chg": 9.0},
        {"code": "000104", "name": "momentum-leader-2", "price": 10.0,
         "amount": 90.0, "pct_chg": 8.0},
        {"code": "000105", "name": "pullback", "price": 10.0,
         "amount": 300.0, "pct_chg": -1.0},
        {"code": "000106", "name": "pullback-2", "price": 10.0,
         "amount": 290.0, "pct_chg": -2.0},
    ])
    selected = select_research_history_universe(
        snapshot, max_symbols=8, min_candidate_amount=80.0,
        liquidity_share=0.25, momentum_share=0.25,
    )
    assert len(selected) == 8
    assert selected.code.nunique() == 8
    assert {"000101", "000102"}.issubset(set(selected.code))
    assert {"000103", "000104"}.issubset(set(selected.code))
    assert {"000105", "000106"}.issubset(set(selected.code))


def test_research_history_pool_unlimited_returns_all_valid_quotes():
    snapshot = pd.DataFrame([
        {"code": "000001", "name": "a", "price": 10.0, "amount": 100.0, "pct_chg": 1.0},
        {"code": "000002", "name": "b", "price": 10.0, "amount": 90.0, "pct_chg": -1.0},
    ])
    selected = select_research_history_universe(snapshot, max_symbols=None)
    assert set(selected.code) == {"000001", "000002"}
