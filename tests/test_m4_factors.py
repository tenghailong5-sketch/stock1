import numpy as np
import pandas as pd
import pytest

from src.model.m4_factors import (
    industry_relative_strength, price_structure, rank_factor, rank_factor_by_date, residual_strength,
)


def bars():
    dates = pd.date_range("2026-01-01", periods=25)
    return pd.DataFrame({
        "symbol": ["000001"]*25, "trade_date": dates,
        "open": np.arange(10, 35, dtype=float),
        "high": np.arange(11, 36, dtype=float),
        "low": np.arange(9, 34, dtype=float),
        "close": np.arange(10.5, 35.5, dtype=float),
        "volume": [100.0]*25, "amount": [2e8]*25,
        "available_time": [pd.Timestamp("2026-11-10T08:00:00Z")]*25,
    })


def test_price_structure_computes_core_factors_and_flat_range_missing():
    x = bars()
    x.loc[24, ["open", "high", "low", "close"]] = [34, 34, 34, 34]
    out = price_structure(x, decision_time="2026-11-10T09:00:00Z")
    assert pd.isna(out.CLV.iloc[-1])
    assert pd.notna(out.return_5.iloc[-1])
    assert out.amount_ma20.iloc[-1] == pytest.approx(2e8)


def test_industry_excess_returns():
    available = pd.Timestamp("2026-01-02T08:00:00Z")
    industry = pd.DataFrame({"trade_date": ["2026-01-01"], "industry_code": ["I1"],
                             "return_5": [.08], "return_20": [.15], "available_time": [available]})
    market = pd.DataFrame({"trade_date": ["2026-01-01"],
                           "market_return_5": [.03], "market_return_20": [.05],
                           "available_time": [available]})
    out = industry_relative_strength(industry, market, decision_time="2026-01-02T09:00:00Z")
    assert out.industry_excess_5.iloc[0] == pytest.approx(.05)
    assert out.industry_excess_20.iloc[0] == pytest.approx(.10)


def test_rank_factor_direction_and_missing():
    values = pd.Series([1., 2., 3., np.nan])
    high = rank_factor(values, higher_is_better=True)
    low = rank_factor(values, higher_is_better=False)
    assert high.iloc[2] == 100
    assert high.iloc[0] < high.iloc[1]
    assert low.iloc[0] == 100
    assert low.iloc[2] < low.iloc[1]
    assert pd.isna(high.iloc[3])


def test_pullback_duration_counts_sessions_since_latest_peak():
    x = bars()
    x.loc[22, ["open", "high", "low", "close"]] = [33.0, 33.5, 32.0, 32.0]
    x.loc[23, ["open", "high", "low", "close"]] = [32.0, 32.5, 30.5, 31.0]
    x.loc[24, ["open", "high", "low", "close"]] = [31.0, 31.5, 29.5, 30.0]
    out = price_structure(x, decision_time="2026-11-10T09:00:00Z")
    assert out.pullback_duration.iloc[22] == 0
    assert out.pullback_duration.iloc[23] == 1
    assert out.pullback_duration.iloc[24] == 2


def test_price_structure_point_in_time_cutoff():
    x = bars()
    x.loc[24, "available_time"] = pd.Timestamp("2026-11-10T09:00:00Z")
    out = price_structure(x, decision_time="2026-11-10T08:30:00Z")
    assert len(out) == 24
    with pytest.raises(ValueError):
        price_structure(x, decision_time="2026-11-10T08:30:00")
    x["available_time"] = "2026-11-10 08:00:00"
    with pytest.raises(ValueError):
        price_structure(x, decision_time="2026-11-10T09:00:00Z")


def test_residual_requires_historical_industry_key():
    stock = pd.DataFrame({"symbol": ["000001"], "trade_date": ["2026-01-01"], "return_5": [.1]})
    market = pd.DataFrame({"trade_date": ["2026-01-01"], "market_return": [.02]})
    industry = pd.DataFrame({"trade_date": ["2026-01-01"], "industry_code": ["I1"], "industry_return": [.03]})
    with pytest.raises(ValueError):
        residual_strength(stock, market, industry)



def test_rank_factor_is_scoped_to_each_trade_date():
    frame = pd.DataFrame({
        "trade_date": ["d1", "d1", "d2", "d2"],
        "factor": [1., 2., 100., 200.],
    })
    ranked = rank_factor_by_date(frame, "factor")
    assert ranked.tolist() == [50.0, 100.0, 50.0, 100.0]


def test_industry_relative_strength_rejects_future_and_naive_availability():
    industry = pd.DataFrame({
        "trade_date": ["d1"], "industry_code": ["I1"], "return_5": [.1],
        "return_20": [.2], "available_time": [pd.Timestamp("2026-01-02T10:00:00Z")]
    })
    market = pd.DataFrame({
        "trade_date": ["d1"], "market_return_5": [.01], "market_return_20": [.02],
        "available_time": [pd.Timestamp("2026-01-02T08:00:00Z")]
    })
    out = industry_relative_strength(industry, market, "2026-01-02T09:00:00Z")
    assert out.empty
    industry["available_time"] = "2026-01-02 08:00:00"
    with pytest.raises(ValueError):
        industry_relative_strength(industry, market, "2026-01-02T09:00:00Z")



def test_price_structure_requires_cutoff_and_rejects_duplicate_or_invalid_bars():
    x = bars()
    with pytest.raises(ValueError, match="decision_time"):
        price_structure(x)
    duplicate = pd.concat([x, x.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        price_structure(duplicate, decision_time="2026-11-10T09:00:00Z")
    bad = x.copy()
    bad.loc[0, "high"] = 1.0
    with pytest.raises(ValueError, match="inconsistent OHLC"):
        price_structure(bad, decision_time="2026-11-10T09:00:00Z")
