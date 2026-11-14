import pandas as pd

from src.model.mainline_validation import compare_ignition_thresholds


def test_ignition_threshold_grid_uses_chronological_holdout():
    rows = []
    dates = pd.date_range("2026-01-01", periods=100)
    for i, day in enumerate(dates):
        for j in range(5):
            cycle = i % 30
            strength = (cycle / 100) if j == 0 else -0.01 * j
            share = (0.02 + cycle / 1000) if j == 0 else 0.02
            rows.append({
                "trade_date": day, "industry_code": f"I{j}",
                "industry_return_5d": strength, "market_return_5d": 0.0,
                "turnover": share * 1000, "market_turnover": 1000,
                "expected_industry_count": 5,
                "industry_close": 100 + i * (0.3 if j == 0 else 0.1),
                "market_close": 100 + i * 0.1,
                "mainline_phase": "TRANSITION", "phase_transition": False,
            })
    grid, summary = compare_ignition_thresholds(
        pd.DataFrame(rows), candidates=((0.55, 0.50), (0.80, 0.65)),
        horizon=10, train_fraction=0.60,
    )
    assert len(grid) == 2
    assert summary["test_set_used_for_selection"] is False
    assert summary["train_label_end"] < summary["split_date"]
    assert set(grid.test_events.astype(int)).issubset(set(range(100)))
