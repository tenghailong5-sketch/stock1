import numpy as np
import pandas as pd
import pytest

from src.model.data_layer import (
    assert_historical_industry_membership, build_total_return_index,
    cross_sectional_percentile, industry_relative_strength,
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
    out = build_total_return_index(p,a,complete_symbols={"000001"})
    assert out.total_return_index.isna().all()


def test_verified_cash_dividend_total_return_is_continuous():
    p = pd.DataFrame({"symbol":["000001","000001"],"trade_date":pd.to_datetime(["2026-01-01","2026-01-02"]),
                      "close":[10.,9.]})
    a = pd.DataFrame({"symbol":["000001"],"ex_date":["2026-01-02"],"action_type":["cash_dividend"],
                      "cash_per_share":[1.],"share_factor":[1.],"verified_status":["VERIFIED"]})
    out = build_total_return_index(p,a,complete_symbols={"000001"})
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


def test_industry_relative_strength_is_point_in_time_and_validates_unique_keys():
    industry = pd.DataFrame([
        {"trade_date":"2026-11-09","industry_code":"BK001","return_5":0.05,
         "return_20":0.10,"available_time":"2026-11-10T07:00:00Z"},
        {"trade_date":"2026-11-09","industry_code":"BK001","return_5":0.99,
         "return_20":0.99,"available_time":"2026-11-10T10:00:00Z"},
    ])
    market = pd.DataFrame([
        {"trade_date":"2026-11-09","return_5":0.02,"return_20":0.04,
         "available_time":"2026-11-10T07:00:00Z"}
    ])
    out = industry_relative_strength(industry, market, decision_time="2026-11-10T08:00:00Z")
    assert len(out) == 1
    assert out.iloc[0].rs_industry_5 == pytest.approx(0.03)


def test_industry_relative_strength_rejects_duplicate_known_rows():
    industry = pd.DataFrame([
        {"trade_date":"2026-11-09","industry_code":"BK001","return_5":0.05,
         "return_20":0.10,"available_time":"2026-11-10T07:00:00Z"},
        {"trade_date":"2026-11-09","industry_code":"BK001","return_5":0.06,
         "return_20":0.11,"available_time":"2026-11-10T07:30:00Z"},
    ])
    market = pd.DataFrame([
        {"trade_date":"2026-11-09","return_5":0.02,"return_20":0.04,
         "available_time":"2026-11-10T07:00:00Z"}
    ])
    with pytest.raises(ValueError, match="duplicate"):
        industry_relative_strength(industry, market, decision_time="2026-11-10T08:00:00Z")


def test_industry_membership_rejects_naive_availability_and_bad_interval():
    base = pd.DataFrame([{
        "symbol":"000001","industry_code":"BK001","valid_from":"2026-01-01",
        "valid_to":"2025-12-31","available_time":"2026-01-01T00:00:00Z","source_id":"x"
    }])
    with pytest.raises(ValueError, match="valid_to"):
        assert_historical_industry_membership(base)
    base.loc[0, "valid_to"] = None
    base.loc[0, "available_time"] = "2026-01-01 00:00:00"
    with pytest.raises(ValueError, match="timezone-aware"):
        assert_historical_industry_membership(base, "2026-02-01T00:00:00Z")

 
def test_total_return_requires_explicit_complete_action_coverage():
    p = pd.DataFrame({"symbol":["000001","000001"],"trade_date":pd.to_datetime(["2026-01-01","2026-01-02"]),
                      "close":[10.,11.]})
    a = pd.DataFrame(columns=["symbol","ex_date","action_type","cash_per_share","share_factor","verified_status"])
    out = build_total_return_index(p,a)
    assert out.total_return_index.isna().all()
    assert set(out.corporate_action_status) == {"MISSING_ACTION_COVERAGE"}


def test_total_return_rejects_duplicate_actions_on_same_ex_date():
    p = pd.DataFrame({"symbol":["000001","000001"],"trade_date":pd.to_datetime(["2026-01-01","2026-01-02"]),
                      "close":[10.,11.]})
    a = pd.DataFrame({"symbol":["000001","000001"],"ex_date":["2026-01-02","2026-01-02"],
                      "action_type":["cash_dividend","bonus"],"cash_per_share":[1.,0.],
                      "share_factor":[1.,2.],"verified_status":["VERIFIED","VERIFIED"]})
    out = build_total_return_index(p,a,complete_symbols={"000001"})
    assert out.total_return_index.isna().all()
    assert set(out.corporate_action_status) == {"INVALID_ACTION_LEDGER"}



@pytest.mark.parametrize("ex_date", ["2025-12-31", "2026-01-01", "2026-01-03"])
def test_total_return_rejects_actions_without_prior_price_coverage(ex_date):
    p = pd.DataFrame({
        "symbol": ["000001", "000001"],
        "trade_date": pd.to_datetime(["2026-01-01", "2026-01-02"]),
        "close": [10.0, 9.0],
    })
    a = pd.DataFrame({
        "symbol": ["000001"], "ex_date": [ex_date],
        "action_type": ["cash_dividend"], "cash_per_share": [1.0],
        "share_factor": [1.0], "verified_status": ["VERIFIED"],
    })
    out = build_total_return_index(p, a, complete_symbols={"000001"})
    assert out.total_return_index.isna().all()
    assert set(out.corporate_action_status) == {"ACTION_DATE_OUTSIDE_PRICE_COVERAGE"}


def test_industry_membership_rejects_naive_availability_without_pit_cutoff():
    frame = pd.DataFrame([{
        "symbol": "000001", "industry_code": "BK001",
        "valid_from": "2026-01-01", "valid_to": None,
        "available_time": "2026-01-01 08:00:00", "source_id": "x",
    }])
    with pytest.raises(ValueError, match="timezone-aware"):
        assert_historical_industry_membership(frame)


@pytest.mark.parametrize("column", ["symbol", "source_id", "source_revision"])
def test_daily_data_rejects_blank_identity_or_source_lineage(column):
    x = sample_bars()
    x.loc[0, column] = "  "
    clean, quality = validate_daily_bars(x, "2026-11-10T10:00:00Z")
    assert clean.empty
    assert not quality.accepted
    assert quality.reason == f"invalid:{column}"


def test_daily_data_rejects_future_trade_date_even_if_provider_claims_available():
    x = sample_bars()
    x.loc[0, "trade_date"] = "2026-11-11"
    clean, quality = validate_daily_bars(x, "2026-11-10T10:00:00Z")
    assert clean.empty
    assert not quality.accepted
    assert quality.reason == "future_trade_date"


def test_daily_data_rejects_ambiguous_latest_source_revision():
    x = sample_bars()
    revision = x.iloc[[0]].copy()
    revision["source_revision"] = "2"
    x = pd.concat([x, revision], ignore_index=True)
    clean, quality = validate_daily_bars(x, "2026-11-10T10:00:00Z")
    assert clean.empty
    assert not quality.accepted
    assert quality.reason == "ambiguous_source_revision"


def test_daily_data_rejects_multiple_providers_for_same_symbol_session():
    x = sample_bars()
    other = x.iloc[[0]].copy()
    other["source_id"] = "other-provider"
    other["source_revision"] = "other-v1"
    x = pd.concat([x, other], ignore_index=True)
    clean, quality = validate_daily_bars(x, "2026-11-10T10:00:00Z")
    assert clean.empty
    assert not quality.accepted
    assert quality.reason == "multiple_sources_for_symbol_trade_date"


def test_daily_data_selects_latest_visible_revision_from_same_provider():
    x = sample_bars()
    revision = x.iloc[[0]].copy()
    revision["available_time"] = "2026-11-09T08:30:00Z"
    revision["source_revision"] = "2"
    revision["close"] = 10.6
    x = pd.concat([x, revision], ignore_index=True)
    clean, quality = validate_daily_bars(x, "2026-11-10T08:30:00Z")
    assert quality.accepted
    assert len(clean) == 1
    assert clean.iloc[0].close == pytest.approx(10.6)
    assert clean.iloc[0].source_revision == "2"
