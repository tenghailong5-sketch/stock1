import numpy as np
import pandas as pd
import pytest

from src.model.m4_relative_strength import (
    rank_industry_excess,
    rank_stock_residual_strength,
)


DECISION = "2026-11-11T10:00:00Z"


def with_lineage(frame):
    out = frame.copy()
    out["source_id"] = "fixture"
    out["source_revision"] = "fixture-v1"
    return out


def test_industry_excess_is_ranked_separately_for_five_and_twenty_sessions():
    date = pd.Timestamp("2026-11-10")
    available = pd.Timestamp(DECISION)
    industries = pd.DataFrame({
        "trade_date": [date] * 3,
        "industry_code": ["I1", "I2", "I3"],
        "return_5": [0.08, 0.04, -0.01],
        "return_20": [0.15, 0.02, -0.03],
        "available_time": [available] * 3,
    })
    market = pd.DataFrame({
        "trade_date": [date],
        "market_return_5": [0.02],
        "market_return_20": [0.01],
        "available_time": [available],
    })
    ranked, audit = rank_industry_excess(
        with_lineage(industries), with_lineage(market), decision_time=DECISION, expected_industry_count=3
    )
    ranked = ranked.set_index("symbol")
    assert ranked.loc["I1", "industry_excess_5_pct"] == 100.0
    assert ranked.loc["I3", "industry_excess_20_pct"] == pytest.approx(33.3333333333)
    assert set(audit["factor"]) == {"industry_excess_5_pct", "industry_excess_20_pct"}
    assert set(audit["status"]) == {"ranked"}


def test_industry_ranks_fail_closed_when_declared_peer_universe_is_incomplete():
    date = pd.Timestamp("2026-11-10")
    available = pd.Timestamp(DECISION)
    industries = pd.DataFrame({
        "trade_date": [date] * 2,
        "industry_code": ["I1", "I2"],
        "return_5": [0.08, 0.04],
        "return_20": [0.15, 0.02],
        "available_time": [available] * 2,
    })
    market = pd.DataFrame({
        "trade_date": [date],
        "market_return_5": [0.02],
        "market_return_20": [0.01],
        "available_time": [available],
    })
    ranked, audit = rank_industry_excess(
        with_lineage(industries), with_lineage(market), decision_time=DECISION, expected_industry_count=3
    )
    assert ranked.filter(regex="_pct$").isna().all().all()
    assert set(audit["status"]) == {"universe_size_mismatch"}


def test_residual_strength_uses_prior_daily_returns_and_audits_stock_pool():
    dates = pd.bdate_range(end="2026-11-10", periods=61)
    available = pd.Timestamp(DECISION)
    market = pd.DataFrame({
        "trade_date": dates,
        "market_return": np.linspace(-0.002, 0.002, len(dates)),
        "market_return_5": [0.01] * len(dates),
        "available_time": [available] * len(dates),
    })
    industry_rows = []
    for code, scale in (("I1", 1.0), ("I2", 1.5)):
        for date in dates:
            industry_rows.append({
                "trade_date": date, "industry_code": code,
                "industry_return": 0.0005 * scale + 0.0002 * np.sin(dates.get_loc(date) * 0.7),
                "industry_return_5": 0.012 * scale,
                "available_time": available,
            })
    industries = pd.DataFrame(industry_rows)
    stock_rows = []
    for i, symbol in enumerate(("000001", "000002", "000003", "000004")):
        code = "I1" if i < 2 else "I2"
        for j, date in enumerate(dates):
            stock_rows.append({
                "symbol": symbol, "industry_code": code, "trade_date": date,
                "daily_return": 0.0002 * (i + 1) + 0.00001 * j,
                "return_5": 0.015 + i * 0.005,
                "available_time": available,
            })
    stocks = pd.DataFrame(stock_rows)
    ranked, audit = rank_stock_residual_strength(
        with_lineage(stocks), with_lineage(market), with_lineage(industries), decision_time=DECISION,
        expected_stock_count=4, window=60, min_periods=40,
    )
    latest = ranked[ranked["trade_date"] == dates[-1]]
    assert len(latest) == 4
    assert latest["residual_return_5"].notna().all()
    assert latest["residual_return_5_pct"].notna().all()
    latest_audit = audit[audit["trade_date"] == dates[-1]]
    assert len(latest_audit) == 1
    assert latest_audit.iloc[0]["status"] == "ranked"


def test_future_available_industry_rows_are_excluded_from_decision():
    date = pd.Timestamp("2026-11-10")
    industries = pd.DataFrame({
        "trade_date": [date],
        "industry_code": ["I1"],
        "return_5": [0.08],
        "return_20": [0.15],
        "available_time": [pd.Timestamp("2026-11-12T00:00:00Z")],
    })
    market = pd.DataFrame({
        "trade_date": [date],
        "market_return_5": [0.02],
        "market_return_20": [0.01],
        "available_time": [pd.Timestamp("2026-11-12T00:00:00Z")],
    })
    ranked, audit = rank_industry_excess(
        with_lineage(industries), with_lineage(market), decision_time=DECISION, expected_industry_count=1
    )
    assert ranked["industry_excess_5_pct"].isna().all()
    assert set(audit["status"]) == {"no_eligible_rows"}
