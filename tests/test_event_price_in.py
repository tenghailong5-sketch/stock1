import pandas as pd

from src.model.event_price_in import evaluate_event_price_in


def _history(daily_returns, *, industry="电子", start="2026-01-01"):
    dates = pd.bdate_range(start, periods=len(daily_returns))
    return pd.DataFrame({
        "trade_date": dates,
        "industry_name": industry,
        "industry_return_1d": daily_returns,
        "market_return_1d": [0.0] * len(dates),
        "turnover": [150.0] * len(dates),
        "market_turnover": [1000.0] * len(dates),
        "observation_available_time": [pd.Timestamp("2026-03-01T10:00:00Z")] * len(dates),
    })


def test_event_price_in_uses_first_seen_clock_and_reports_pricing_in():
    history = _history([0.0] * 19 + [0.10, 0.025, 0.02, 0.01], start="2026-01-01")
    history.loc[history.trade_date > pd.Timestamp("2026-01-28"), "turnover"] = 300.0
    event = {
        "candidate_id": "event-1",
        "first_seen_at": "2026-01-28T09:00:00+00:00",
        "mapped_industries": ["电子"],
        "published_at": "2026-01-01T00:00:00Z",
    }
    result, audit = evaluate_event_price_in(
        [event], history,
        decision_time=pd.Timestamp("2026-03-02T00:00:00Z"),
    )
    row = result.iloc[0]
    assert row.price_in_state == "PRICING_IN"
    assert row.post_discovery_sessions_observed == 3
    assert abs(row.pre_discovery_excess_20d) < 1e-12  # same-day bar is excluded
    assert not bool(row.impact_direction_inferred)
    assert not bool(row.historical_pit_eligible)
    assert audit["state_counts"]["PRICING_IN"] == 1


def test_event_price_in_never_uses_same_or_pre_first_seen_session_as_post():
    history = _history([0.02] * 25, start="2026-01-01")
    event = {
        "candidate_id": "event-2",
        "first_seen_at": "2026-02-05T09:00:00+00:00",
        "mapped_industries": ["电子"],
    }
    result, _ = evaluate_event_price_in(
        [event], history,
        decision_time=pd.Timestamp("2026-03-02T00:00:00Z"),
    )
    row = result.iloc[0]
    assert row.post_discovery_sessions_observed == 0
    assert row.price_in_state == "AWAITING_POST_EVENT_SESSION"
    assert row.reason == "no_post_event_trading_session_observed"


def test_event_price_in_unknown_when_mapping_or_clock_missing():
    history = _history([0.0] * 25)
    result, _ = evaluate_event_price_in(
        [{"candidate_id": "event-3", "first_seen_at": None}],
        history,
        decision_time=pd.Timestamp("2026-03-02T00:00:00Z"),
    )
    assert result.iloc[0].price_in_state == "UNKNOWN"


def test_event_price_in_unknown_when_post_session_has_no_valid_return():
    history = _history([0.0] * 27, start="2026-01-01")
    history.loc[history.trade_date == pd.Timestamp("2026-02-06"), "industry_return_1d"] = float("nan")
    event = {
        "candidate_id": "event-4",
        "first_seen_at": "2026-02-05T09:00:00+00:00",
        "mapped_industries": ["电子"],
    }
    result, _ = evaluate_event_price_in(
        [event], history,
        decision_time=pd.Timestamp("2026-03-02T00:00:00Z"),
    )
    assert result.iloc[0].price_in_state == "UNKNOWN"
    assert result.iloc[0].reason == "post_event_return_unavailable"
