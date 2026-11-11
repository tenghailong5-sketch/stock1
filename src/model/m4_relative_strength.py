"""Audited relative-strength percentile adapters for FHM-100 M4.

This module deliberately preserves 5-session and 20-session industry excess
returns as separate sub-factors. It does not invent a composite formula or
pretend retrieval-time historical bars are historically point-in-time data.
"""
from __future__ import annotations

import pandas as pd

from .m4_factors import industry_relative_strength, residual_strength
from .m4_percentile import audited_percentile_rank


def _rank_one(
    frame: pd.DataFrame,
    raw_col: str,
    output_col: str,
    *,
    scope_id: str,
    expected_universe_size: int,
    min_coverage: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ranked = frame.copy()
    if raw_col not in ranked.columns:
        ranked[output_col] = float("nan")
        return ranked, pd.DataFrame([{
            "trade_date": None,
            "factor": output_col,
            "scope_id": scope_id,
            "expected_universe_size": expected_universe_size,
            "observed_universe_size": 0,
            "symbol_membership_sha256": None,
            "finite_factor_count": 0,
            "factor_coverage": 0.0,
            "min_coverage": min_coverage,
            "status": "missing_raw_factor",
        }])
    ranks, audit = audited_percentile_rank(
        ranked,
        raw_col,
        scope_id=scope_id,
        expected_universe_size=expected_universe_size,
        higher_is_better=True,
        min_coverage=min_coverage,
        symbol_col="industry_code" if "industry_code" in ranked and "symbol" not in ranked else "symbol",
    )
    ranked[output_col] = ranks
    if audit.empty:
        audit = pd.DataFrame([{
            "trade_date": None,
            "scope_id": scope_id,
            "expected_universe_size": expected_universe_size,
            "observed_universe_size": 0,
            "symbol_membership_sha256": None,
            "finite_factor_count": 0,
            "factor_coverage": 0.0,
            "min_coverage": min_coverage,
            "status": "no_eligible_rows",
        }])
    audit["factor"] = output_col
    return ranked, audit


def rank_industry_excess(
    industry_returns: pd.DataFrame,
    market_returns: pd.DataFrame,
    *,
    decision_time,
    expected_industry_count: int,
    scope_id: str = "archived_industry_universe",
    min_coverage: float = 0.95,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute 5d/20d industry excess returns and rank within industries."""
    raw = industry_relative_strength(
        industry_returns, market_returns, decision_time=decision_time
    )
    # The industry identifier is the peer identity for this cross-section.
    ranked = raw.rename(columns={"industry_code": "symbol"})
    ranked, audit5 = _rank_one(
        ranked, "industry_excess_5", "industry_excess_5_pct",
        scope_id=scope_id, expected_universe_size=expected_industry_count,
        min_coverage=min_coverage,
    )
    ranked, audit20 = _rank_one(
        ranked, "industry_excess_20", "industry_excess_20_pct",
        scope_id=scope_id, expected_universe_size=expected_industry_count,
        min_coverage=min_coverage,
    )
    audit = pd.concat([audit5, audit20], ignore_index=True)
    return ranked, audit


def rank_stock_residual_strength(
    stock_returns: pd.DataFrame,
    market_returns: pd.DataFrame,
    industry_returns: pd.DataFrame,
    *,
    decision_time,
    expected_stock_count: int,
    window: int = 60,
    min_periods: int = 40,
    scope_id: str = "top_350_turnover_universe",
    min_coverage: float = 0.95,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute PIT residual returns, then rank stocks within the stock pool."""
    raw = residual_strength(
        stock_returns,
        market_returns,
        industry_returns,
        window=window,
        min_periods=min_periods,
        decision_time=decision_time,
    )
    ranked, audit = _rank_one(
        raw,
        "residual_return_5",
        "residual_return_5_pct",
        scope_id=scope_id,
        expected_universe_size=expected_stock_count,
        min_coverage=min_coverage,
    )
    return ranked, audit
