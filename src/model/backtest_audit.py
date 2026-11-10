"""Point-in-time backtest admission audit.

This module deliberately separates data readiness from performance simulation.
It refuses historical backtests when observations were only retrieved later,
or when survivorship/corporate-action/execution evidence is incomplete.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Iterable

import pandas as pd


@dataclass(frozen=True)
class BacktestAudit:
    ready: bool
    candidate_rows: int
    bar_rows: int
    errors: tuple[str, ...]

    def to_dict(self) -> dict:
        result = asdict(self)
        result["errors"] = list(self.errors)
        return result


_CANDIDATE_REQUIRED = {
    "symbol", "decision_time", "available_time", "historical_pit_eligible",
    "source_revision", "strategy", "score", "eligible", "corporate_actions_resolved",
    "universe_complete", "delisted_included", "ruleset_version",
}
_BAR_REQUIRED = {
    "symbol", "trade_date", "available_time", "historical_pit_eligible",
    "source_revision", "open", "high", "low", "close", "volume",
    "suspended", "limit_up_locked", "limit_down_locked",
    "corporate_actions_resolved", "universe_complete", "delisted_included",
}


def _aware_timestamp(value) -> bool:
    try:
        ts = pd.Timestamp(value)
        return not pd.isna(ts) and ts.tzinfo is not None
    except (TypeError, ValueError, OverflowError):
        return False


def audit_backtest_dataset(candidates: pd.DataFrame, bars: pd.DataFrame) -> BacktestAudit:
    """Audit required PIT and survivorship evidence; never calculates returns.

    Candidate availability must precede its decision timestamp. Daily bars
    must be explicitly certified as historical PIT observations. A later
    retrieval timestamp cannot be relabelled as historical availability.
    """
    errors: list[str] = []
    missing_candidates = sorted(_CANDIDATE_REQUIRED - set(candidates.columns))
    missing_bars = sorted(_BAR_REQUIRED - set(bars.columns))
    if missing_candidates:
        errors.append("candidate_missing_columns:" + ",".join(missing_candidates))
    if missing_bars:
        errors.append("bar_missing_columns:" + ",".join(missing_bars))
    if missing_candidates or missing_bars:
        return BacktestAudit(False, len(candidates), len(bars), tuple(errors))

    if candidates.empty:
        errors.append("candidate_dataset_empty")
    if bars.empty:
        errors.append("bar_dataset_empty")

    for label, frame in (("candidate", candidates), ("bar", bars)):
        if frame[["symbol", "source_revision"]].isna().any().any():
            errors.append(label + "_missing_identity_or_revision")
        if frame.duplicated(["symbol", "source_revision"]).any():
            errors.append(label + "_duplicate_symbol_revision")
        if not frame["historical_pit_eligible"].map(lambda x: isinstance(x, bool) and x).all():
            errors.append(label + "_contains_non_pit_observations")
        for col in ("corporate_actions_resolved", "universe_complete", "delisted_included"):
            if not frame[col].map(lambda x: isinstance(x, bool) and x).all():
                errors.append(label + "_not_certified:" + col)
        if not frame["available_time"].map(_aware_timestamp).all():
            errors.append(label + "_invalid_available_time")

    if not candidates["decision_time"].map(_aware_timestamp).all():
        errors.append("candidate_invalid_decision_time")
    else:
        available = pd.to_datetime(candidates["available_time"], utc=True, errors="coerce")
        decisions = pd.to_datetime(candidates["decision_time"], utc=True, errors="coerce")
        if (available > decisions).any():
            errors.append("candidate_lookahead_available_after_decision")

    if not bars["trade_date"].map(_aware_timestamp).all():
        errors.append("bar_trade_date_must_be_timezone_aware")
    numeric_cols = ["open", "high", "low", "close", "volume"]
    numeric = bars[numeric_cols].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any():
        errors.append("bar_contains_missing_or_non_numeric_ohlcv")
    else:
        if (numeric[["open", "high", "low", "close"]] <= 0).any().any():
            errors.append("bar_contains_nonpositive_price")
        if (numeric["volume"] < 0).any():
            errors.append("bar_contains_negative_volume")
        if (numeric["high"] < numeric[["open", "low", "close"]].max(axis=1)).any():
            errors.append("bar_invalid_high")
        if (numeric["low"] > numeric[["open", "high", "close"]].min(axis=1)).any():
            errors.append("bar_invalid_low")

    if not candidates["eligible"].map(lambda x: isinstance(x, bool)).all():
        errors.append("candidate_eligibility_must_be_boolean")
    eligible = candidates.loc[candidates["eligible"] == True]  # noqa: E712
    scores = pd.to_numeric(eligible["score"], errors="coerce")
    if scores.isna().any() or ((scores < 80) | (scores > 100)).any():
        errors.append("eligible_candidate_score_outside_80_100")

    if candidates["ruleset_version"].isna().any():
        errors.append("candidate_missing_ruleset_version")

    return BacktestAudit(not errors, len(candidates), len(bars), tuple(errors))


def require_backtest_ready(candidates: pd.DataFrame, bars: pd.DataFrame) -> BacktestAudit:
    """Return the audit or raise; intentionally no performance metric fallback."""
    audit = audit_backtest_dataset(candidates, bars)
    if not audit.ready:
        raise ValueError("historical backtest blocked: " + ";".join(audit.errors))
    return audit
