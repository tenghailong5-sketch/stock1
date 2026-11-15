import pandas as pd
import pytest

from src.model.mainline_flows import classify_mainlines


def _panel(observed=20, expected=20):
    rows = []
    dates = pd.bdate_range("2026-01-05", periods=8)
    for day_i, day in enumerate(dates):
        for industry_i in range(observed):
            rows.append({
                "trade_date": day,
                "industry_code": f"I{industry_i:02d}",
                "industry_return_3d": 0.02 + industry_i * 0.0001 + day_i * 0.0002,
                "market_return_3d": 0.01,
                "industry_return_5d": 0.03 + industry_i * 0.0001 + day_i * 0.0003,
                "market_return_5d": 0.015,
                "turnover": 1000 + industry_i * 100,
                "market_turnover": sum(1000 + i * 100 for i in range(observed)),
                "expected_industry_count": expected,
            })
    return pd.DataFrame(rows)


def test_mainline_accepts_ninety_percent_industry_coverage():
    result = classify_mainlines(_panel(observed=18, expected=20))
    assert len(result) == 18 * 8
    assert result["strength_percentile"].notna().all()


def test_mainline_rejects_coverage_below_ninety_percent():
    with pytest.raises(ValueError, match="coverage below configured minimum"):
        classify_mainlines(_panel(observed=17, expected=20))


def test_mainline_can_use_stricter_coverage_when_requested():
    with pytest.raises(ValueError, match="coverage below configured minimum"):
        classify_mainlines(
            _panel(observed=18, expected=20), min_industry_coverage=0.95
        )
