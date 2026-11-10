import numpy as np
import pandas as pd
import pytest

from src.model.m4_factors import (
    industry_relative_strength, price_structure, rank_factor, residual_strength,
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
    out = price_structure(x)
    assert pd.isna(out.CLV.iloc[-1])
    assert pd.notna(out.return_5.iloc[-1])
    assert out.amount_ma20.iloc[-1] == pytest.approx(2e8)


def test_industry_excess_returns():
    industry = pd.DataFrame({"trade_date": ["2026-01-01"], "industry_code": ["I1"],
                             "return_5": [.08], "return_20": [.15]})
    market = pd.DataFrame({"trade_date": ["2026-01-01"],
                           "market_return_5": [.03], "market_return_20": [.05]})
    out = industry_relative_strength(industry, market)
    assert out.industry_excess_5.iloc[0] == pytest.approx(.05)
    assert out.industry_excess_20.iloc[0] == pytest.approx(.10)


def test_rank_factor_preserves_missing():
    out = rank_factor(pd.Series([1., 2., np.nan]))
    assert out.iloc[1] == 100
    assert pd.isna(out.iloc[2])


def test_residual_requires_historical_industry_key():
    stock = pd.DataFrame({"symbol": ["000001"], "trade_date": ["2026-01-01"], "return_5": [.1]})
    market = pd.DataFrame({"trade_date": ["2026-01-01"], "market_return": [.02]})
    industry = pd.DataFrame({"trade_date": ["2026-01-01"], "industry_code": ["I1"], "industry_return": [.03]})
    with pytest.raises(ValueError):
        residual_strength(stock, market, industry)
