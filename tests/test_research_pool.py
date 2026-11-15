import pandas as pd
import pytest

from src.fhm100 import select_research_history_universe


def test_research_pool_keeps_financials_observable_and_prioritizes_active_names():
    snapshot = pd.DataFrame([
        {"code": "600000", "name": "浦发银行", "price": 10, "amount": 9_000_000_000, "pct_chg": 2},
        {"code": "601398", "name": "工商银行", "price": 8, "amount": 8_000_000_000, "pct_chg": 1},
        {"code": "300001", "name": "成长科技A", "price": 20, "amount": 2_000_000_000, "pct_chg": 5},
        {"code": "300002", "name": "成长科技B", "price": 20, "amount": 1_500_000_000, "pct_chg": 4},
        {"code": "300003", "name": "成长科技C", "price": 20, "amount": 1_000_000_000, "pct_chg": -2},
    ])

    selected = select_research_history_universe(
        snapshot, max_symbols=3, min_candidate_amount=100_000_000,
        liquidity_share=1 / 3,
    )

    assert selected["code"].tolist() == ["600000", "300001", "300002"]
    assert "600000" in set(selected["code"])


def test_research_pool_includes_limit_up_momentum_without_chase_authorization():
    snapshot = pd.DataFrame([
        {"code": "300001", "name": "涨停科技", "price": 20, "amount": 2_000_000_000, "pct_chg": 10},
        {"code": "300002", "name": "温和成长", "price": 20, "amount": 1_000_000_000, "pct_chg": 3},
        {"code": "300003", "name": "活跃制造", "price": 20, "amount": 800_000_000, "pct_chg": 0},
        {"code": "300004", "name": "回撤成长", "price": 20, "amount": 700_000_000, "pct_chg": -1},
    ])

    selected = select_research_history_universe(
        snapshot, max_symbols=3, min_candidate_amount=100_000_000,
        liquidity_share=1 / 3,
    )

    assert "300001" in selected["code"].tolist()
    assert len(selected) == 3
    assert selected["code"].is_unique


def test_research_pool_validates_arguments():
    empty = pd.DataFrame(columns=["code", "name", "price", "amount", "pct_chg"])
    with pytest.raises(ValueError, match="max_symbols"):
        select_research_history_universe(empty, max_symbols=0)
    with pytest.raises(ValueError, match="liquidity_share"):
        select_research_history_universe(empty, liquidity_share=0)
