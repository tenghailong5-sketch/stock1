import numpy as np
import pandas as pd
import pytest

from src.model.m4_factors import (
    build_m4_pit_factor_snapshot, industry_relative_strength, price_structure, rank_factor, rank_factor_by_date, residual_strength,
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
        "source_id": ["fixture"]*25, "source_revision": ["fixture-v1"]*25,
    })


def with_lineage(frame):
    out = frame.copy()
    out["source_id"] = "fixture"
    out["source_revision"] = "fixture-v1"
    return out


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
    out = industry_relative_strength(with_lineage(industry), with_lineage(market), decision_time="2026-01-02T09:00:00Z")
    assert out.industry_excess_5.iloc[0] == pytest.approx(.05)
    assert out.industry_excess_20.iloc[0] == pytest.approx(.10)



def test_build_m4_snapshot_ranks_only_complete_declared_universe():
    dates = pd.bdate_range("2026-01-01", periods=45)
    available = pd.Timestamp("2026-03-10T08:00:00Z")
    symbols = ["000001", "000002"]
    bars_rows, stock_rows = [], []
    for symbol, offset, industry_code in zip(symbols, [0.0, 0.01], ["I1", "I2"]):
        for i, date in enumerate(dates):
            close = 10.0 + i * (0.1 + offset)
            bars_rows.append({
                "symbol": symbol, "trade_date": date, "open": close - 0.05,
                "high": close + 0.1, "low": close - 0.1, "close": close,
                "volume": 1000.0 + i, "amount": 2e8 + i * 1e6,
                "available_time": available, "source_id": "bars", "source_revision": "v1",
                "availability_basis": "original_publication_time", "historical_pit_eligible": True,
                "pit_total_return_close": close, "pit_total_return_high": close + 0.1,
                "pit_total_return_available_time": available, "corporate_action_status": "VERIFIED",
            })
            stock_rows.append({
                "symbol": symbol, "trade_date": date, "industry_code": industry_code,
                "pit_total_return_return_5": 0.02 + offset,
                "pit_total_return_daily_return": 0.001 + offset / 10,
                "pit_total_return_available_time": available, "corporate_action_status": "VERIFIED",
                "available_time": available, "source_id": "stock", "source_revision": "v1",
                "availability_basis": "original_publication_time", "historical_pit_eligible": True,
            })
    market = pd.DataFrame({
        "trade_date": dates, "market_return": np.linspace(-0.001, 0.001, len(dates)),
        "market_return_5": [0.01] * len(dates), "market_return_20": [0.02] * len(dates),
        "available_time": available, "source_id": "market", "source_revision": "v1",
        "availability_basis": "original_publication_time", "historical_pit_eligible": True,
    })
    industry = pd.DataFrame([
        {"trade_date": date, "industry_code": code, "industry_return": 0.002 + j * 0.001,
         "industry_return_5": 0.02 + j * 0.01, "return_5": 0.02 + j * 0.01,
         "return_20": 0.04 + j * 0.01, "available_time": available,
         "source_id": "industry", "source_revision": "v1",
         "availability_basis": "original_publication_time", "historical_pit_eligible": True}
        for date in dates for j, code in enumerate(["I1", "I2"])
    ])
    # The residual regression needs independently identifiable benchmark series.
    industry["industry_return"] = np.tile(np.sin(np.arange(len(dates)) * 0.7) * 0.001, 2)
    out, audit = build_m4_pit_factor_snapshot(
        pd.DataFrame(bars_rows), pd.DataFrame(stock_rows), industry, market,
        decision_time="2026-03-10T09:00:00+08:00", expected_universe_size=2,
    )
    assert len(out) == 2
    assert out["industry_relative_strength_pct"].notna().all()
    assert out["price_structure_pct"].notna().all()
    assert out["stock_residual_strength_pct"].notna().any()
    assert out["historical_pit_eligible"].all()
    assert audit.loc[audit.factor_name.eq("price_structure_pct"), "status"].iloc[0] == "ranked"
    assert audit.loc[audit.factor_name.eq("pullback_reversal_structure_pct"), "status"].iloc[0] == "insufficient_factor_coverage"


def test_build_m4_snapshot_blocks_universe_size_mismatch():
    with pytest.raises(ValueError, match="expected_universe_size"):
        build_m4_pit_factor_snapshot(
            pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(),
            decision_time="2026-03-10T09:00:00+08:00", expected_universe_size=0,
        )

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
    out = industry_relative_strength(with_lineage(industry), with_lineage(market), "2026-01-02T09:00:00Z")
    assert out.empty
    industry["available_time"] = "2026-01-02 08:00:00"
    with pytest.raises(ValueError):
        industry_relative_strength(with_lineage(industry), with_lineage(market), "2026-01-02T09:00:00Z")



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
        industry_relative_strength(with_lineage(industry), with_lineage(market), "2026-01-02T09:00:00Z")


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
        industry_relative_strength(with_lineage(industry), with_lineage(market), "2026-01-02T09:00:00Z")


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
        "pit_total_return_daily_return": stock_daily,
        "pit_total_return_return_5": [0.02] * len(dates),
        "pit_total_return_available_time": [available] * len(dates),
        "corporate_action_status": ["VERIFIED"] * len(dates),
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
        with_lineage(stocks), with_lineage(market), with_lineage(industry), window=60, min_periods=40,
        decision_time="2026-03-10T11:00:00Z",
    )
    # R2.1 defines residual return using only the two benchmark betas.
    # The fitted intercept is a regression nuisance term, not a third
    # benchmark component to subtract from the target 5-session return.
    assert out["residual_return_5"].iloc[-1] == pytest.approx(
        0.02 - beta_market * 0.01 - beta_industry * 0.002,
        abs=1e-8,
    )


def test_price_structure_requires_source_lineage():
    x = bars().drop(columns=["source_revision"])
    with pytest.raises(ValueError, match="missing"):
        price_structure(x, decision_time="2026-11-10T09:00:00Z")


def test_industry_relative_strength_requires_source_lineage():
    available = pd.Timestamp("2026-01-02T08:00:00Z")
    industry = pd.DataFrame({
        "trade_date": ["2026-01-01"], "industry_code": ["I1"],
        "return_5": [.08], "return_20": [.15], "available_time": [available],
        "source_id": ["provider"], "source_revision": ["v1"],
    })
    market = pd.DataFrame({
        "trade_date": ["2026-01-01"], "market_return_5": [.03],
        "market_return_20": [.05], "available_time": [available],
        "source_id": ["provider"],
    })
    with pytest.raises(ValueError, match="source_revision"):
        industry_relative_strength(industry, market, decision_time="2026-01-02T09:00:00Z")


def test_industry_relative_strength_rejects_blank_source_lineage():
    available = pd.Timestamp("2026-01-02T08:00:00Z")
    industry = pd.DataFrame({
        "trade_date": ["2026-01-01"], "industry_code": ["I1"],
        "return_5": [.08], "return_20": [.15], "available_time": [available],
        "source_id": ["provider"], "source_revision": ["  "],
    })
    market = pd.DataFrame({
        "trade_date": ["2026-01-01"], "market_return_5": [.03],
        "market_return_20": [.05], "available_time": [available],
        "source_id": ["provider"], "source_revision": ["v1"],
    })
    with pytest.raises(ValueError, match="invalid source_revision"):
        industry_relative_strength(industry, market, decision_time="2026-01-02T09:00:00Z")


def test_residual_strength_rejects_rank_deficient_market_industry_regression():
    dates = pd.bdate_range(end="2026-11-10", periods=61)
    available = pd.Timestamp("2026-11-11T10:00:00Z")
    market_values = np.linspace(-0.002, 0.002, len(dates))
    market = pd.DataFrame({
        "trade_date": dates,
        "market_return": market_values,
        "market_return_5": [0.01] * len(dates),
        "available_time": [available] * len(dates),
    })
    industries = pd.DataFrame({
        "trade_date": dates,
        "industry_code": ["I1"] * len(dates),
        "industry_return": market_values,
        "industry_return_5": [0.012] * len(dates),
        "available_time": [available] * len(dates),
    })
    stocks = pd.DataFrame({
        "symbol": ["000001"] * len(dates),
        "industry_code": ["I1"] * len(dates),
        "trade_date": dates,
        "pit_total_return_daily_return": np.linspace(-0.001, 0.003, len(dates)),
        "pit_total_return_return_5": [0.025] * len(dates),
        "pit_total_return_available_time": [available] * len(dates),
        "corporate_action_status": ["VERIFIED"] * len(dates),
        "available_time": [available] * len(dates),
    })
    result = residual_strength(
        with_lineage(stocks), with_lineage(market), with_lineage(industries),
        window=60, min_periods=40,
        decision_time="2026-11-11T10:00:00Z",
    )
    assert result["residual_return_5"].isna().all()

def test_industry_relative_strength_normalizes_numeric_strings_and_keeps_invalid_values_missing():
    available = pd.Timestamp("2026-01-02T08:00:00Z")
    industry = pd.DataFrame({
        "trade_date": ["2026-01-01", "2026-01-01"],
        "industry_code": ["I1", "I2"],
        "return_5": ["0.08", "not-a-number"],
        "return_20": ["0.15", "0.20"],
        "available_time": [available, available],
    })
    market = pd.DataFrame({
        "trade_date": ["2026-01-01"],
        "market_return_5": ["0.03"],
        "market_return_20": ["0.05"],
        "available_time": [available],
    })
    result = industry_relative_strength(
        with_lineage(industry), with_lineage(market),
        decision_time="2026-01-02T09:00:00Z",
    ).set_index("industry_code")
    assert result.loc["I1", "industry_excess_5"] == pytest.approx(0.05)
    assert result.loc["I1", "industry_excess_20"] == pytest.approx(0.10)
    assert pd.isna(result.loc["I2", "industry_excess_5"])
    assert result.loc["I2", "industry_excess_20"] == pytest.approx(0.15)


def test_industry_relative_strength_rejects_infinite_numeric_observations():
    available = pd.Timestamp("2026-01-02T08:00:00Z")
    industry = pd.DataFrame({
        "trade_date": ["2026-01-01"], "industry_code": ["I1"],
        "return_5": [np.inf], "return_20": [0.15],
        "available_time": [available],
    })
    market = pd.DataFrame({
        "trade_date": ["2026-01-01"], "market_return_5": [0.03],
        "market_return_20": [0.05], "available_time": [available],
    })
    result = industry_relative_strength(
        with_lineage(industry), with_lineage(market),
        decision_time="2026-01-02T09:00:00Z",
    )
    assert pd.isna(result.loc[0, "industry_excess_5"])
    assert result.loc[0, "industry_excess_20"] == pytest.approx(0.10)


def test_cross_session_price_volume_flags_remain_missing_without_verified_pit_prices():
    out = price_structure(
        bars(), decision_time="2026-11-10T09:00:00Z"
    )
    for column in (
        "price_up_volume_confirmation", "high_volume_stall", "low_volume_pullback"
    ):
        assert out[column].isna().all()


def test_cross_session_price_volume_flags_use_verified_total_return_close():
    x = bars()
    x["pit_total_return_close"] = 100.0 + np.arange(len(x)) * 0.5
    x["pit_total_return_high"] = x["pit_total_return_close"] + 0.8
    x["pit_total_return_available_time"] = pd.Timestamp("2026-11-10T08:00:00Z")
    x["corporate_action_status"] = "VERIFIED"
    # Make the final session genuinely higher-volume than its 20-session baseline.
    x.loc[x.index[-1], "amount"] = 3e8
    out = price_structure(x, decision_time="2026-11-10T09:00:00Z")
    assert out["price_up_volume_confirmation"].iloc[-1] == True
    assert out["high_volume_stall"].iloc[-1] == False
    assert out["low_volume_pullback"].iloc[-1] == False


def test_cross_session_price_volume_flags_stay_missing_for_unverified_actions():
    x = bars()
    x["pit_total_return_close"] = 100.0 + np.arange(len(x)) * 0.5
    x["pit_total_return_high"] = x["pit_total_return_close"] + 0.8
    x["pit_total_return_available_time"] = pd.Timestamp("2026-11-10T08:00:00Z")
    x["corporate_action_status"] = "PENDING"
    out = price_structure(x, decision_time="2026-11-10T09:00:00Z")
    for column in (
        "price_up_volume_confirmation", "high_volume_stall", "low_volume_pullback"
    ):
        assert out[column].isna().all()

def test_price_volume_flags_preserve_missing_rolling_baselines_as_na():
    x = bars()
    x["pit_total_return_close"] = 100.0 + np.arange(len(x)) * 0.5
    x["pit_total_return_high"] = x["pit_total_return_close"] + 0.8
    x["pit_total_return_available_time"] = pd.Timestamp("2026-11-10T08:00:00Z")
    x["corporate_action_status"] = "VERIFIED"
    out = price_structure(x, decision_time="2026-11-10T09:00:00Z")
    # The 20-session amount baseline is unavailable for the first 19 rows;
    # the 5-session volume baseline is unavailable for the first 4 rows.
    assert out["price_up_volume_confirmation"].iloc[:19].isna().all()
    assert out["high_volume_stall"].iloc[:19].isna().all()
    assert out["low_volume_pullback"].iloc[:4].isna().all()


def test_price_structure_zero_activity_windows_do_not_create_infinite_ratios():
    x = bars()
    x["amount"] = 0.0
    x["volume"] = 0.0
    x["pit_total_return_close"] = 100.0 + np.arange(len(x)) * 0.5
    x["pit_total_return_high"] = x["pit_total_return_close"] + 0.8
    x["pit_total_return_available_time"] = pd.Timestamp("2026-11-10T08:00:00Z")
    x["corporate_action_status"] = "VERIFIED"
    out = price_structure(x, decision_time="2026-11-10T09:00:00Z")
    for column in ("LQ", "amount_ma5_to_ma20", "volume_ma5_to_ma20"):
        assert out[column].isna().all()
    assert out["price_up_volume_confirmation"].isna().all()
    assert out["high_volume_stall"].isna().all()

def test_rank_factor_keeps_infinite_and_malformed_values_missing():
    values = pd.Series([1.0, 2.0, np.inf, -np.inf, "not-a-number"])
    ranked = rank_factor(values)
    assert ranked.iloc[0] == pytest.approx(50.0)
    assert ranked.iloc[1] == pytest.approx(100.0)
    assert ranked.iloc[2:].isna().all()


def test_industry_relative_strength_returns_only_latest_decision_snapshot():
    available = pd.Timestamp("2026-01-05T08:00:00Z")
    industry = pd.DataFrame({
        "trade_date": ["2026-01-02", "2026-01-05"],
        "industry_code": ["I1", "I1"],
        "return_5": [0.04, 0.06],
        "return_20": [0.08, 0.12],
        "available_time": [available, available],
    })
    market = pd.DataFrame({
        "trade_date": ["2026-01-02", "2026-01-05"],
        "market_return_5": [0.01, 0.02],
        "market_return_20": [0.03, 0.04],
        "available_time": [available, available],
    })
    result = industry_relative_strength(
        with_lineage(industry), with_lineage(market),
        decision_time="2026-01-05T09:00:00Z",
    )
    assert result["trade_date"].dt.strftime("%Y-%m-%d").tolist() == ["2026-01-05"]
    assert result["industry_excess_5"].iloc[0] == pytest.approx(0.04)


def test_residual_strength_returns_only_latest_decision_snapshot():
    dates = pd.bdate_range("2026-01-01", periods=45)
    available = pd.Timestamp("2026-03-10T10:00:00Z")
    market_daily = np.sin(np.arange(len(dates)) * 0.3) * 0.001
    industry_daily = np.cos(np.arange(len(dates)) * 0.5) * 0.001
    stock_daily = 0.2 * market_daily + 0.8 * industry_daily + 0.0001
    stocks = pd.DataFrame({
        "symbol": ["000001"] * len(dates), "industry_code": ["I1"] * len(dates),
        "trade_date": dates, "pit_total_return_daily_return": stock_daily,
        "pit_total_return_return_5": [0.01] * len(dates),
        "pit_total_return_available_time": [available] * len(dates),
        "corporate_action_status": ["VERIFIED"] * len(dates),
        "available_time": [available] * len(dates),
    })
    market = pd.DataFrame({
        "trade_date": dates, "market_return": market_daily,
        "market_return_5": [0.002] * len(dates), "available_time": [available] * len(dates),
    })
    industry = pd.DataFrame({
        "trade_date": dates, "industry_code": ["I1"] * len(dates),
        "industry_return": industry_daily, "industry_return_5": [0.003] * len(dates),
        "available_time": [available] * len(dates),
    })
    result = residual_strength(
        with_lineage(stocks), with_lineage(market), with_lineage(industry),
        window=60, min_periods=40, decision_time="2026-03-10T11:00:00Z",
    )
    assert len(result) == 1
    assert pd.Timestamp(result["trade_date"].iloc[0]) == dates[-1]


def test_residual_strength_rejects_unverified_corporate_action_series():
    dates = pd.bdate_range("2026-01-01", periods=45)
    available = pd.Timestamp("2026-03-10T10:00:00Z")
    daily = np.sin(np.arange(len(dates)) * 0.2) * 0.001
    stocks = pd.DataFrame({
        "symbol": ["000001"] * len(dates), "industry_code": ["I1"] * len(dates),
        "trade_date": dates, "pit_total_return_daily_return": daily,
        "pit_total_return_return_5": [0.01] * len(dates),
        "pit_total_return_available_time": [available] * len(dates),
        "corporate_action_status": ["PENDING"] * len(dates),
        "available_time": [available] * len(dates),
    })
    market = pd.DataFrame({
        "trade_date": dates, "market_return": daily * 0.5,
        "market_return_5": [0.002] * len(dates), "available_time": [available] * len(dates),
    })
    industry = pd.DataFrame({
        "trade_date": dates, "industry_code": ["I1"] * len(dates),
        "industry_return": daily * 0.3, "industry_return_5": [0.003] * len(dates),
        "available_time": [available] * len(dates),
    })
    result = residual_strength(
        with_lineage(stocks), with_lineage(market), with_lineage(industry),
        window=60, min_periods=40, decision_time="2026-03-10T11:00:00Z",
    )
    assert result.empty


def test_build_m4_snapshot_rejects_retrieval_time_only_history_before_calculation():
    frame = pd.DataFrame({
        "historical_pit_eligible": [False],
        "availability_basis": ["retrieval_time_only"],
        "available_time": [pd.Timestamp("2026-03-10T08:00:00Z")],
        "source_id": ["akshare"], "source_revision": ["v1"],
    })
    with pytest.raises(ValueError, match="not certified historical PIT eligible"):
        from src.model.m4_factors import _require_historical_pit_eligible
        _require_historical_pit_eligible(frame, "stock_bars")


def test_build_m4_snapshot_rejects_missing_pit_eligibility_metadata():
    frame = pd.DataFrame({
        "available_time": [pd.Timestamp("2026-03-10T08:00:00Z")],
        "source_id": ["provider"], "source_revision": ["v1"],
    })
    with pytest.raises(ValueError, match="missing PIT eligibility fields"):
        from src.model.m4_factors import _require_historical_pit_eligible
        _require_historical_pit_eligible(frame, "market_returns")
