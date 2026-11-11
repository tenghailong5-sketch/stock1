"""Audited cross-sectional percentile ranks with explicit universe coverage.

These ranks are diagnostic sub-factor percentiles only. They are not M5
strategy scores and must not be interpreted outside the declared peer universe.
"""
from __future__ import annotations

import hashlib
import math

import pandas as pd

from .m4_factors import rank_factor


def audited_percentile_rank(
    frame: pd.DataFrame,
    value_col: str,
    *,
    scope_id: str,
    expected_universe_size: int,
    symbol_col: str = "symbol",
    date_col: str = "trade_date",
    higher_is_better: bool = True,
    min_coverage: float = 0.95,
) -> tuple[pd.Series, pd.DataFrame]:
    """Return per-date ranks plus an audit table; fail closed on incomplete pools.

    scope_id names the peer-universe definition (for example,
    top_350_turnover_universe). The SHA-256 membership fingerprint records the
    exact sorted symbol set actually ranked on each date; it does not claim to
    validate that set against an external constituent master.
    """
    if not isinstance(scope_id, str) or not scope_id.strip():
        raise ValueError("scope_id must be a non-empty universe identifier")
    if isinstance(expected_universe_size, bool) or not isinstance(expected_universe_size, int):
        raise ValueError("expected_universe_size must be a positive integer")
    if expected_universe_size <= 0:
        raise ValueError("expected_universe_size must be a positive integer")
    try:
        coverage_floor = float(min_coverage)
    except (TypeError, ValueError, OverflowError):
        raise ValueError("min_coverage must be finite and within (0, 1]") from None
    if not math.isfinite(coverage_floor) or not 0 < coverage_floor <= 1:
        raise ValueError("min_coverage must be finite and within (0, 1]")

    required = {symbol_col, date_col, value_col}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError("missing columns: " + ",".join(sorted(missing)))
    if frame[symbol_col].isna().any() or frame[date_col].isna().any():
        raise ValueError("symbol/date cannot be missing")
    if frame.duplicated([date_col, symbol_col]).any():
        raise ValueError("duplicate symbol within decision date")

    numeric = pd.to_numeric(frame[value_col], errors="coerce")
    numeric = numeric.where(numeric.map(lambda x: pd.notna(x) and math.isfinite(float(x))))
    ranks = pd.Series(float("nan"), index=frame.index, dtype="float64", name=f"{value_col}_percentile")
    audits = []

    for date_value, indexes in frame.groupby(date_col, sort=False).groups.items():
        symbols = sorted(str(value) for value in frame.loc[indexes, symbol_col].tolist())
        membership_fingerprint = hashlib.sha256("\n".join(symbols).encode("utf-8")).hexdigest()
        observed = len(symbols)
        valid = numeric.loc[indexes].notna()
        finite_count = int(valid.sum())
        coverage = finite_count / expected_universe_size
        if observed != expected_universe_size:
            status = "universe_size_mismatch"
        elif coverage < coverage_floor:
            status = "insufficient_factor_coverage"
        else:
            status = "ranked"
            ranks.loc[indexes] = rank_factor(numeric.loc[indexes], higher_is_better=higher_is_better)
        audits.append({
            "trade_date": date_value,
            "scope_id": scope_id,
            "expected_universe_size": expected_universe_size,
            "observed_universe_size": observed,
            "symbol_membership_sha256": membership_fingerprint,
            "finite_factor_count": finite_count,
            "factor_coverage": coverage,
            "min_coverage": coverage_floor,
            "status": status,
        })
    return ranks, pd.DataFrame(audits)
