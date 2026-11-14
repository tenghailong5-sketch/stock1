"""Strict forward-only assembly of M4 stock/market/industry return inputs.

The current membership snapshot is applied only for a prospective decision.
It is not a historical point-in-time membership history and must never be
used to claim unbiased historical backtest results.
"""
from __future__ import annotations

import pandas as pd

from .m4_archive_adapter import (
    build_industry_index_returns,
    build_market_index_returns,
    build_stock_daily_returns,
)
from .m4_industry_observations import select_latest_observed_industry_snapshot


def build_forward_m4_inputs(
    stock_bars: pd.DataFrame,
    market_bars: pd.DataFrame,
    industry_bars: pd.DataFrame,
    membership: pd.DataFrame,
    *,
    expected_symbols,
    benchmark_symbol: str,
    decision_time,
    max_membership_age="2D",
    min_stock_coverage: float = 1.0,
) -> dict:
    """Assemble forward M4 frames; incomplete or ambiguous inputs fail closed."""
    cutoff = pd.Timestamp(decision_time)
    if cutoff.tzinfo is None:
        raise ValueError("decision_time must be timezone-aware")
    if not benchmark_symbol or not str(benchmark_symbol).strip():
        raise ValueError("benchmark_symbol must be explicit")
    if not 0 < float(min_stock_coverage) <= 1:
        raise ValueError("min_stock_coverage must be in (0, 1]")

    symbols = [str(value).strip() for value in expected_symbols]
    if not symbols or len(symbols) != len(set(symbols)) or any(not s for s in symbols):
        raise ValueError("expected_symbols must be unique non-empty symbols")

    selected, membership_audit = select_latest_observed_industry_snapshot(
        membership, symbols, decision_time=cutoff, max_age=max_membership_age
    )
    if membership_audit["status"] != "ready_forward_only":
        raise ValueError("membership snapshot blocked: " + membership_audit["status"])

    # This adapter supports one current industry assignment per stock. Do not
    # silently choose one when the taxonomy source returns multiple memberships.
    if selected["symbol"].duplicated().any():
        raise ValueError("multiple current industry assignments per stock are ambiguous")
    selected["symbol"] = selected["symbol"].astype(str).str.strip()
    selected["industry_code"] = selected["industry_code"].astype(str).str.strip()
    membership_time = pd.to_datetime(selected["available_time"], utc=True).max()
    if membership_time > cutoff.tz_convert("UTC"):
        raise ValueError("membership snapshot was not available at decision_time")

    stock = build_stock_daily_returns(stock_bars, decision_time=cutoff)
    market_input = market_bars.loc[
        market_bars["index_symbol"].astype(str).str.strip() == str(benchmark_symbol).strip()
    ].copy()
    if market_input.empty:
        raise ValueError("explicit market benchmark is absent: " + str(benchmark_symbol))
    market = build_market_index_returns(market_input, decision_time=cutoff)
    industry = build_industry_index_returns(industry_bars, decision_time=cutoff)

    stock_counts = stock.groupby("symbol")["trade_date"].nunique()
    required_symbols = set(symbols)
    available_symbols = set(stock_counts.index.astype(str))
    missing_symbols = required_symbols - available_symbols
    if missing_symbols:
        raise ValueError("stock archive missing expected symbols: " + ",".join(sorted(missing_symbols)))
    # At least 41 derived rows provide 40 prior daily observations plus a 5-session
    # target return for the default residual model's minimum regression sample.
    short_history = sorted(s for s in symbols if int(stock_counts.loc[s]) < 41)
    if short_history:
        raise ValueError("insufficient stock history for residual model: " + ",".join(short_history))

    mapping = selected[["symbol", "industry_code"]].copy()
    stock = stock.loc[stock["symbol"].isin(required_symbols)].merge(
        mapping, on="symbol", how="left", validate="many_to_one"
    )
    if stock["industry_code"].isna().any():
        raise ValueError("one or more expected stocks lack industry mapping")

    required_codes = set(mapping["industry_code"])
    available_codes = set(industry["industry_code"].astype(str))
    missing_codes = required_codes - available_codes
    if missing_codes:
        raise ValueError("industry index history missing mapped codes: " + ",".join(sorted(missing_codes)))

    industry = industry.loc[industry["industry_code"].astype(str).isin(required_codes)].copy()
    if industry.duplicated(["trade_date", "industry_code"]).any():
        raise ValueError("duplicate industry index date/code rows")
    if market.duplicated(["trade_date"]).any():
        raise ValueError("duplicate market benchmark dates")

    # All output rows become visible no earlier than the current membership
    # observation. This is deliberately forward-only, not historical PIT.
    stock["available_time"] = pd.to_datetime(stock["available_time"], utc=True).where(
        pd.to_datetime(stock["available_time"], utc=True) >= membership_time, membership_time
    )
    stock["historical_pit_eligible"] = False
    stock["membership_snapshot_id"] = membership_audit["snapshot_id"]
    stock["membership_observed_at"] = pd.to_datetime(selected["observed_at"], utc=True).max()

    for name, frame, keys in (
        ("stock", stock, ["symbol", "trade_date"]),
        ("market", market, ["trade_date"]),
        ("industry", industry, ["industry_code", "trade_date"]),
    ):
        if frame.duplicated(keys).any():
            raise ValueError("duplicate " + name + " rows after forward M4 assembly")

    # Require a date-aligned benchmark and industry observation for every stock
    # return row; no zero-fill, forward-fill, or silent row dropping.
    check = stock[["symbol", "trade_date", "industry_code"]].merge(
        market[["trade_date"]], on="trade_date", how="left", indicator="_market"
    ).merge(
        industry[["trade_date", "industry_code"]], on=["trade_date", "industry_code"],
        how="left", indicator="_industry"
    )
    if not check["_market"].eq("both").all():
        raise ValueError("market benchmark dates do not cover all stock return dates")
    if not check["_industry"].eq("both").all():
        raise ValueError("industry index dates do not cover all stock return dates")

    coverage = len(required_symbols & set(stock["symbol"])) / len(required_symbols)
    if coverage < float(min_stock_coverage):
        raise ValueError("stock universe coverage below configured threshold")

    return {
        "stock_returns": stock.sort_values(["symbol", "trade_date"]).reset_index(drop=True),
        "market_returns": market.sort_values("trade_date").reset_index(drop=True),
        "industry_returns": industry.sort_values(["industry_code", "trade_date"]).reset_index(drop=True),
        "membership_audit": membership_audit,
        "stock_universe_size": len(required_symbols),
        "stock_coverage": coverage,
        "benchmark_symbol": str(benchmark_symbol).strip(),
        "historical_pit_eligible": False,
        "status": "ready_forward_only_diagnostic",
    }
