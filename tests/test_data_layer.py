import numpy as np
import pandas as pd
import pytest

from src.model.data_layer import (
    build_total_return_index, cross_sectional_percentile,
    rolling_residual_strength, validate_daily_bars,
)


def sample_bars():
    return pd.DataFrame([
        {"symbol":"000001","trade_date":"2026-11-09","open":10,"high":11,"low":9,"close":10.5,
         "volume":100,"amount":1000,"available_time":"2026-11-09T08:00:00Z","source_id":"x","source_revision":"1"},
        {"symbol":"000001","trade_date":"2026-11-10","open":10.5,"high":11,"low":10,"close":10.8,
         "volume":100,"amount":1100,"available_time":"2026-11-10T09:00:00Z","source_id":"x","source_revision":"1"},
    ])


def test_daily_data_filters_future_availability():
    x, q = validate_daily_bars(sample_bars(), "2026-11-10T08:30:00Z")
    assert q.accepted
    assert len(x) == 1


def test_daily_data_rejects_inconsistent_ohlc():
    x = sample_bars()
    x.loc[0, "high"] = 8
    clean, _ = validate_daily_bars(x, "2026-11-11T00:00:00Z")
    assert len(clean) == 1
    assert clean.iloc[0].trade_date == pd.Timestamp("2026-11-10")


def test_unverified_corporate_action_blocks_total_return():
    p = pd.DataFrame({"symbol":["000001","000001"],"trade_date":pd.to_datetime(["2026-01-01","2026-01-02"]),
                      "close":[10.,9.]})
    a = pd.DataFrame({"symbol":["000001"],"ex_date":["2026-01-02"],"action_type":["cash_dividend"],
                      "cash_per_share":[1.],"share_factor":[1.],"verified_status":["PENDING"]})
    out = build_total_return_index(p,a)
    assert out.total_return_index.isna().all()


def test_verified_cash_dividend_total_return_is_continuous():
    p = pd.DataFrame({"symbol":["000001","000001"],"trade_date":pd.to_datetime(["2026-01-01","2026-01-02"]),
                      "close":[10.,9.]})
    a = pd.DataFrame({"symbol":["000001"],"ex_date":["2026-01-02"],"action_type":["cash_dividend"],
                      "cash_per_share":[1.],"share_factor":[1.],"verified_status":["VERIFIED"]})
    out = build_total_return_index(p,a)
    assert out.total_return_index.iloc[-1] == pytest.approx(100.0)


def test_residual_uses_prior_window_only():
    idx = pd.date_range("2026-01-01", periods=50)
    m = pd.Series(np.linspace(-.01,.01,50), index=idx)
    ind = pd.Series(np.linspace(.01,-.01,50), index=idx)
    stock = 1.2*m + .8*ind
    resid = rolling_residual_strength(stock, m, ind, window=45, min_periods=30)
    assert resid.iloc[-1] == pytest.approx(0, abs=1e-10)


def test_percentile_preserves_missing():
    s = cross_sectional_percentile(pd.Series([1.,2.,np.nan]))
    assert pd.isna(s.iloc[2])
    assert s.iloc[1] > s.iloc[0]



def test_daily_data_rejects_timezone_naive_availability():
    x = sample_bars()
    x["available_time"] = ["2026-11-09 08:00:00", "2026-11-10 09:00:00"]
    clean, quality = validate_daily_bars(x, "2026-11-11T00:00:00Z")
    assert clean.empty
    assert not quality.accepted
    assert quality.reason == "available_time_must_be_timezone_aware"
