import pandas as pd

from src.fhm100 import merge_current_quotes_with_history


def test_current_quote_universe_survives_missing_history_and_keeps_live_values():
    snapshot = pd.DataFrame({
        "code": ["000001", "000002"],
        "name": ["A", "B"],
        "price": [10.5, 20.5],
        "amount": [300_000_000, 100_000_000],
        "pct_chg": [3.2, -1.5],
    })
    history = pd.DataFrame({
        "代码": ["000001"],
        "名称": ["stale-name"],
        "最新价": [9.9],
        "涨跌幅%": [1.0],
        "当日成交额": [50_000_000],
        "symbol": ["000001"],
        "available_time": ["2026-11-14T00:00:00+00:00"],
        "CLV": [0.8],
        "技术观察通过": [True],
        "流动性观察通过": [True],
    })
    result = merge_current_quotes_with_history(
        snapshot, history, observed_at=pd.Timestamp("2026-11-15T10:00:00Z")
    ).set_index("代码")

    assert len(result) == 2
    assert result.loc["000001", "最新价"] == 10.5
    assert result.loc["000001", "涨跌幅%"] == 3.2
    assert result.loc["000001", "当日成交额"] == 300_000_000
    assert result.loc["000001", "CLV"] == 0.8
    assert bool(result.loc["000001", "技术观察通过"])
    assert bool(result.loc["000001", "history_data_available"])
    assert pd.isna(result.loc["000002", "CLV"])
    assert not bool(result.loc["000002", "history_data_available"])
    assert result.loc["000002", "最新价"] == 20.5
