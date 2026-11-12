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
    assert pd.isna(out.return_5.iloc[-1])
    assert out.cross_session_price_status.iloc[-1] == "MISSING_PIT_TOTAL_RETURN_FIELDS"
    assert out.amount_ma20.iloc[-1] == pytest.approx(2e8)



def test_price_structure_uses_only_verified_pit_total_return_series_for_cross_session_features():
    x = bars()
    x["pit_total_return_close"] = 100.0 + np.arange(len(x)) * 0.5
    x["pit_total_return_high"] = x["pit_total_return_close"] + 0.8
    x["pit_total_return_available_time"] = pd.Timestamp("2026-11-10T08:00:00Z")
    x["corporate_action_status"] = "VERIFIED"
    out = price_structure(x, decision_time="2026-11-10T09:00:00Z")
    expected = (out.pit_total_return_close.iloc[-1]
                / out.pit_total_return_close.iloc[-6]) - 1
    assert out.cross_session_price_status.eq("READY").all()
    assert out.return_5.iloc[-1] == pytest.approx(expected)
    assert pd.notna(out.breakout20.iloc[-1])
    assert pd.notna(out.Pullback.iloc[-1])


def test_price_structure_disables_cross_session_features_when_action_ledger_unverified():
    x = bars()
    x["pit_total_return_close"] = 100.0 + np.arange(len(x)) * 0.5
    x["pit_total_return_high"] = x["pit_total_return_close"] + 0.8
    x["pit_total_return_available_time"] = pd.Timestamp("2026-11-10T08:00:00Z")
    x["corporate_action_status"] = "PENDING"
    out = price_structure(x, decision_time="2026-11-10T09:00:00Z")
    assert out.cross_session_price_status.eq("UNVERIFIED_CORPORATE_ACTIONS").all()
    assert out.return_5.isna().all()
    assert out.breakout20.isna().all()
    assert out.Pullback.isna().all()



def test_price_structure_rejects_pit_total_return_series_available_after_cutoff():
    x = bars()
    x["pit_total_return_close"] = 100.0 + np.arange(len(x)) * 0.5
    x["pit_total_return_high"] = x["pit_total_return_close"] + 0.8
    x["pit_total_return_available_time"] = pd.Timestamp("2026-11-10T09:30:00Z")
    x["corporate_action_status"] = "VERIFIED"
    out = price_structure(x, decision_time="2026-11-10T09:00:00Z")
    assert out.cross_session_price_status.eq(
        "PIT_TOTAL_RETURN_NOT_AVAILABLE_AT_DECISION"
    ).all()
    assert out.return_5.isna().all()




def test_price_structure_disables_cross_session_features_for_naive_pit_availability():
    x = bars()
    x["pit_total_return_close"] = 100.0 + np.arange(len(x)) * 0.5
    x["pit_total_return_high"] = x["pit_total_return_close"] + 0.8
    x["pit_total_return_available_time"] = "2026-11-10 08:00:00"
    x["corporate_action_status"] = "VERIFIED"
    out = price_structure(x, decision_time="2026-11-10T09:00:00Z")
    assert out.cross_session_price_status.eq(
        "PIT_TOTAL_RETURN_NOT_AVAILABLE_AT_DECISION"
    ).all()
    assert out.return_5.isna().all()



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
    x["pit_total_return_close"] = x["close"]
    x["pit_total_return_high"] = x["high"]
    x["pit_total_return_available_time"] = pd.Timestamp("2026-11-10T08:00:00Z")
    x["corporate_action_status"] = "VERIFIED"
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
        "trade_date": ["2026-01-02"], "industry_code": ["I1"], "return_5": [.1],
        "return_20": [.2], "available_time": [pd.Timestamp("2026-01-02T10:00:00Z")]
    })
    market = pd.DataFrame({
        "trade_date": ["2026-01-02"], "market_return_5": [.01], "market_return_20": [.02],
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



@pytest.mark.parametrize("column,value,error", [
    ("volume", -1.0, "negative volume or amount"),
    ("amount", -1.0, "negative volume or amount"),
    ("amount", np.nan, "invalid OHLCV/amount"),
    ("volume", np.inf, "non-finite OHLCV/amount"),
])
def test_price_structure_rejects_invalid_volume_and_amount(column, value, error):
    x = bars()
    x.loc[24, column] = value
    with pytest.raises(ValueError, match=error):
        price_structure(x, decision_time="2026-11-10T09:00:00Z")


def test_price_structure_rejects_future_trade_date_even_if_timestamp_claims_available():
    x = bars()
    x.loc[24, "trade_date"] = pd.Timestamp("2026-11-11")
    x.loc[24, "available_time"] = pd.Timestamp("2026-11-10T08:00:00Z")
    with pytest.raises(ValueError, match="future trade_date"):
        price_structure(x, decision_time="2026-11-10T09:00:00Z")


def test_price_structure_rejects_blank_symbol():
    x = bars()
    x.loc[0, "symbol"] = "  "
    with pytest.raises(ValueError, match="invalid symbol"):
        price_structure(x, decision_time="2026-11-10T09:00:00Z")



def test_industry_relative_strength_rejects_future_trade_date_with_forged_available_time():
    available = pd.Timestamp("2026-01-02T08:00:00Z")
    industry = pd.DataFrame({
        "trade_date": ["2026-01-03"], "industry_code": ["I1"],
        "return_5": [.1], "return_20": [.2], "available_time": [available],
    })
    market = pd.DataFrame({
        "trade_date": ["2026-01-02"], "market_return_5": [.01],
        "market_return_20": [.02], "available_time": [available],
    })
    with pytest.raises(ValueError, match="future trade_date"):
        industry_relative_strength(industry, market, "2026-01-02T09:00:00Z")


def test_industry_relative_strength_rejects_blank_historical_industry_code():
    available = pd.Timestamp("2026-01-02T08:00:00Z")
    industry = pd.DataFrame({
        "trade_date": ["2026-01-02"], "industry_code": ["  "],
        "return_5": [.1], "return_20": [.2], "available_time": [available],
    })
    market = pd.DataFrame({
        "trade_date": ["2026-01-02"], "market_return_5": [.01],
        "market_return_20": [.02], "available_time": [available],
    })
    with pytest.raises(ValueError, match="invalid industry_code"):
        industry_relative_strength(industry, market, "2026-01-02T09:00:00Z")


def test_residual_strength_matches_r21_formula_without_subtracting_regression_intercept():
    dates = pd.bdate_range("2026-01-01", periods=45)
    market_daily = np.linspace(-0.002, 0.002, len(dates))
    industry_daily = np.sin(np.arange(len(dates)) * 0.7) * 0.001
    alpha, beta_market, beta_industry = 0.0003, 1.7, 0.8
    stock_daily = alpha + beta_market * market_daily + beta_industry * industry_daily
    available = pd.Timestamp("2026-03-10T10:00:00Z")
    stocks = pd.DataFrame({
        "symbol": ["000001"] * len(dates),
        "industry_code": ["I1"] * len(dates),
        "trade_date": dates,
        "daily_return": stock_daily,
        "return_5": [0.02] * len(dates),
        "available_time": [available] * len(dates),
    })
    market = pd.DataFrame({
        "trade_date": dates,
        "market_return": market_daily,
        "market_return_5": [0.01] * len(dates),
        "available_time": [available] * len(dates),
    })
    industry = pd.DataFrame({
        "trade_date": dates,
        "industry_code": ["I1"] * len(dates),
        "industry_return": industry_daily,
        "industry_return_5": [0.002] * len(dates),
        "available_time": [available] * len(dates),
    })
    out = residual_strength(
        stocks, market, industry, window=60, min_periods=40,
        decision_time="2026-03-10T11:00:00Z",
    )
    # R2.1 defines residual return using only the two benchmark betas.
    # The fitted intercept is a regression nuisance term, not a third
    # benchmark component to subtract from the target 5-session return.
    assert out["residual_return_5"].iloc[-1] == pytest.approx(
        0.02 - beta_market * 0.01 - beta_industry * 0.002,
        abs=1e-8,
    )
