import pandas as pd
import pytest

from src.model.market_state import (
    apply_state_cooldown, calculate_state_scores, classify_raw_state,
    market_environment_score,
)


def history(n=65):
    return pd.DataFrame({
        "trade_date": pd.date_range("2026-01-01", periods=n),
        "close": [11.0]*n, "ma20": [10.0]*n, "ma60": [9.0]*n,
        "breadth": [0.65]*n, "amount": [120.0]*n, "amount_ma20": [100.0]*n,
        "limit_up_cnt": [30]*n, "limit_down_cnt": [5]*n, "vol20": [0.02]*n,
        "available_time": [pd.Timestamp("2026-11-10T08:00:00Z")]*n,
    })


def test_m3_requires_full_60_session_percentile_history():
    out = calculate_state_scores(history(59), decision_time="2026-11-10T09:00:00Z")
    assert out.StateScore.isna().all()
    out = calculate_state_scores(history(60), decision_time="2026-11-10T09:00:00Z")
    assert pd.notna(out.StateScore.iloc[-1])


def test_m3_score_matches_specification():
    out = calculate_state_scores(history(60), decision_time="2026-11-10T09:00:00Z")
    row = out.iloc[-1]
    expected = .35*1 + .25*1 + .20*1 + .20*1 - .10*1
    assert row.StateScore == pytest.approx(expected)


def test_risk_off_precedes_risk_on():
    row = {"StateScore": .8, "S_trend": 1, "S_limit": .1,
           "limit_up_cnt": 20, "limit_down_cnt": 2}
    assert classify_raw_state(row) == "RISK_OFF"


def test_state_switch_requires_two_consecutive_sessions_and_applies_next_day():
    x = pd.DataFrame({
        "trade_date": pd.date_range("2026-01-01", periods=5),
        "raw_market_state": ["RISK_ON", "RISK_ON", "RISK_OFF", "RISK_ON", "RISK_ON"],
    })
    out = apply_state_cooldown(x)
    assert out.market_state.tolist() == ["NEUTRAL", "NEUTRAL", "RISK_ON", "RISK_ON", "RISK_ON"]


def test_market_score_missing_state_is_nan():
    assert pd.isna(market_environment_score({"StateScore": .7, "market_state": "INSUFFICIENT_DATA"}))


def test_risk_on_market_mapping_reaches_100_at_max_attainable_score():
    assert market_environment_score({"StateScore": .9, "market_state": "RISK_ON"}) == pytest.approx(100)


def test_neutral_market_mapping_uses_neutral_interval():
    assert market_environment_score({"StateScore": .475, "market_state": "NEUTRAL"}) == pytest.approx(57.5)



def test_m3_requires_point_in_time_and_excludes_future_market_rows():
    x = history(60)
    x.loc[59, "available_time"] = pd.Timestamp("2026-11-10T10:00:00Z")
    out = calculate_state_scores(x, decision_time="2026-11-10T09:00:00Z")
    assert len(out) == 59
    with pytest.raises(ValueError, match="timezone-aware"):
        calculate_state_scores(x, decision_time="2026-11-10T09:00:00")


def test_m3_missing_trend_inputs_do_not_become_false_bearish_scores():
    x = history(60)
    x.loc[59, "close"] = float("nan")
    out = calculate_state_scores(x, decision_time="2026-11-10T09:00:00Z")
    assert pd.isna(out.iloc[-1].S_trend)
    assert pd.isna(out.iloc[-1].StateScore)
