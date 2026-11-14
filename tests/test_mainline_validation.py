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



def test_frozen_chronological_holdout_uses_later_dates_without_tuning():
    from src.model.mainline_validation import evaluate_frozen_chronological_holdout

    rows = []
    dates = pd.date_range("2026-01-01", periods=60)
    for code in ("I1", "I2"):
        for i, day in enumerate(dates):
            transition = i in (3, 45)
            rows.append({
                "trade_date": day,
                "industry_code": code,
                "mainline_phase": "IGNITION" if transition else "TRANSITION",
                "phase_transition": transition,
                "industry_close": 100 + i * (1 if code == "I1" else -0.2),
                "market_close": 100 + i * 0.1,
            })
    events, summary = evaluate_frozen_chronological_holdout(
        pd.DataFrame(rows), horizons=(5, 10), train_fraction=0.6
    )
    assert summary["split_date"] == dates[36].date().isoformat()
    assert summary["train_label_end"] == dates[25].date().isoformat()
    assert set(events.validation_partition) == {"TRAIN", "HOLDOUT"}
    assert events.loc[events.validation_partition.eq("TRAIN"), "trade_date"].max() <= pd.Timestamp(summary["train_label_end"])
    assert events.loc[events.validation_partition.eq("HOLDOUT"), "trade_date"].min() >= pd.Timestamp(summary["split_date"])
    assert summary["classifier_frozen"] is True
    assert summary["threshold_search_used"] is False
    assert summary["holdout_used_for_selection"] is False
    assert summary["historical_pit_eligible"] is False


def test_frozen_holdout_rejects_empty_or_nonpositive_horizons():
    from src.model.mainline_validation import evaluate_frozen_chronological_holdout

    dates = pd.date_range("2026-01-01", periods=40)
    frame = pd.DataFrame([{
        "trade_date": day, "industry_code": "I1",
        "mainline_phase": "TRANSITION", "phase_transition": False,
        "industry_close": 100 + i, "market_close": 100 + i * 0.1,
    } for i, day in enumerate(dates)])
    for invalid in ((), (0,), (True,)):
        with pytest.raises(ValueError, match="horizons must contain positive integer"):
            evaluate_frozen_chronological_holdout(frame, horizons=invalid)


def test_frozen_holdout_rejects_missing_validation_columns_before_date_access():
    from src.model.mainline_validation import evaluate_frozen_chronological_holdout

    with pytest.raises(ValueError, match="missing validation fields"):
        evaluate_frozen_chronological_holdout(pd.DataFrame({"close": [1]}))
