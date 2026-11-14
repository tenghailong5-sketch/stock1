import pandas as pd
import pytest

from src.model.mainline_validation import evaluate_mainline_transitions


def test_transition_validation_uses_forward_returns_only_as_outcomes():
    rows = []
    dates = pd.date_range("2026-01-01", periods=13)
    for code, start, direction in (("I1", 0, 1), ("I2", 1, -1)):
        for i, day in enumerate(dates):
            rows.append({
                "trade_date": day, "industry_code": code,
                "mainline_phase": "IGNITION" if i == start else "TRANSITION",
                "phase_transition": i == start,
                "industry_close": 100 + i * direction,
                "market_close": 100,
            })
    events, summary = evaluate_mainline_transitions(pd.DataFrame(rows), horizons=(5, 10))
    assert len(events) == 2
    assert summary["IGNITION"]["evaluated_5d"] == 2
    assert summary["IGNITION"]["directional_hit_rate_5d"] == 0.5
    assert events.historical_pit_eligible.eq(False).all()
    assert summary["historical_pit_eligible"] is False


def test_transition_validation_rejects_duplicate_and_missing_inputs():
    frame = pd.DataFrame([{
        "trade_date": "2026-01-01", "industry_code": "I1",
        "mainline_phase": "IGNITION", "phase_transition": True,
        "industry_close": 100, "market_close": 100,
    }])
    with pytest.raises(ValueError, match="duplicate"):
        evaluate_mainline_transitions(pd.concat([frame, frame]), horizons=(1,))
    with pytest.raises(ValueError, match="missing validation fields"):
        evaluate_mainline_transitions(frame.drop(columns=["market_close"]), horizons=(1,))
