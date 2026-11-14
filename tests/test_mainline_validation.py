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


def test_ignition_threshold_grid_uses_chronological_holdout():
    from src.model.mainline_validation import compare_ignition_thresholds

    rows = []
    dates = pd.date_range("2026-01-01", periods=60)
    for day_i, day in enumerate(dates):
        for industry_i in range(20):
            ret = ((industry_i % 7) - 3) * 0.002 + (day_i % 5) * 0.0005
            rows.append({
                "trade_date": day,
                "industry_code": f"I{industry_i:02d}",
                "mainline_phase": "TRANSITION",
                "phase_transition": False,
                "industry_close": 100 + day_i * (industry_i + 1) * 0.01,
                "market_close": 100 + day_i * 0.1,
                "industry_return_3d": ret,
                "market_return_3d": 0.001,
                "industry_return_5d": ret * 1.5,
                "market_return_5d": 0.002,
                "turnover": 1000 + industry_i * 20 + day_i * (industry_i + 1),
                "market_turnover": 30000,
                "expected_industry_count": 20,
            })
    results, summary = compare_ignition_thresholds(
        pd.DataFrame(rows), candidates=((0.5, 0.5), (0.7, 0.6)),
        horizon=5, train_fraction=0.6,
    )
    assert len(results) == 2
    assert results.split_date.nunique() == 1
    assert summary["test_set_used_for_selection"] is False
    assert summary["historical_pit_eligible"] is False
