import pandas as pd

from src.providers.eastmoney_market_emotion import fetch_market_emotion_pool_history


def test_eastmoney_pool_history_derives_board_failure_promotion_and_streak():
    pools = {
        "20261112": {
            "limit_up": pd.DataFrame({"代码": ["A", "B"], "连板数": [1, 2]}),
            "broken": pd.DataFrame({"代码": ["X"]}),
            "strong": pd.DataFrame({"代码": ["S"]}),
        },
        "20261113": {
            "limit_up": pd.DataFrame({"代码": ["A", "B", "C"], "连板数": [2, 3, 1]}),
            "broken": pd.DataFrame({"代码": ["D", "E"]}),
            "strong": pd.DataFrame({"代码": ["T", "U"]}),
        },
    }
    result = fetch_market_emotion_pool_history(
        ["2026-11-12", "2026-11-13"],
        limit_up_fetcher=lambda date: pools[date]["limit_up"],
        broken_board_fetcher=lambda date: pools[date]["broken"],
        strong_pool_fetcher=lambda date: pools[date]["strong"],
        clock=lambda: pd.Timestamp("2026-11-14T00:00:00Z"),
    )
    second = result.iloc[1]
    assert result.pool_history_status.eq("READY").all()
    assert second.highest_observed_limit_up_streak == 3
    assert second.strong_pool_count == 2
    assert second.board_failure_rate == 0.4
    assert second.promotion_rate == 1.0
    assert second.promotion_numerator == 1
    assert second.promotion_denominator == 1
    assert result.historical_pit_eligible.eq(False).all()
    assert result.availability_basis.eq("retrieval_time_only").all()


def test_eastmoney_pool_history_keeps_incomplete_provider_data_missing():
    result = fetch_market_emotion_pool_history(
        ["2026-11-13"],
        limit_up_fetcher=lambda date: pd.DataFrame({"代码": ["A"], "连板数": [1]}),
        broken_board_fetcher=lambda date: (_ for _ in ()).throw(TimeoutError()),
        strong_pool_fetcher=lambda date: pd.DataFrame({"代码": ["S"]}),
        clock=lambda: pd.Timestamp("2026-11-14T00:00:00Z"),
    )
    row = result.iloc[0]
    assert row.pool_history_status == "INSUFFICIENT"
    assert pd.isna(row.board_failure_rate)
    assert row.highest_observed_limit_up_streak == 1
    assert row.strong_pool_count == 1
