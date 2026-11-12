"""Strict point-in-time corporate-action gate and total-return price builder.

This module accepts normalized provider observations. It does not infer complete
coverage from a successful fetch or from an empty event response. A symbol is
eligible only when an explicit, time-stamped COMPLETE coverage attestation and
all event/price observations were available by the decision cutoff.

The output is a normalized wealth index (base 100), not an execution price.
Raw OHLC must remain the source for order placement and same-session geometry.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


PRICE_REQUIRED = {
    "symbol", "trade_date", "open", "high", "low", "close",
    "available_time", "source_id", "source_revision",
}
ACTION_REQUIRED = {
    "symbol", "ex_date", "action_type", "cash_per_share", "share_factor",
    "verified_status", "available_time", "source_id", "source_revision",
}
COVERAGE_REQUIRED = {
    "symbol", "coverage_start", "coverage_end", "coverage_status",
    "available_time", "source_id", "source_revision",
}
SUPPORTED_ACTIONS = {"cash_dividend", "split", "bonus"}


@dataclass(frozen=True)
class PITBuildResult:
    prices: pd.DataFrame
    status: str
    reason: str


def _utc(value, field: str) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        raise ValueError(f"{field} must be timezone-aware")
    return ts.tz_convert("UTC")


def _timestamps(frame: pd.DataFrame, column: str, label: str) -> pd.Series:
    values = pd.to_datetime(frame[column], errors="coerce")
    if values.isna().any() or values.dt.tz is None:
        raise ValueError(f"{label}.{column} must be valid and timezone-aware")
    return values.dt.tz_convert("UTC")


def _blocked(prices: pd.DataFrame, status: str, reason: str) -> PITBuildResult:
    out = prices.copy()
    for col in ("pit_total_return_close", "pit_total_return_high",
                "pit_total_return_available_time"):
        out[col] = pd.NaT if col.endswith("available_time") else np.nan
    out["corporate_action_status"] = status
    out["pit_total_return_reason"] = reason
    return PITBuildResult(out, status, reason)


def build_pit_total_return_prices(
    prices: pd.DataFrame,
    actions: pd.DataFrame,
    coverage: pd.DataFrame,
    *,
    decision_time,
    symbols: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Build per-symbol base-100 total-return close/high series, failing closed.

    Coverage must contain one COMPLETE attestation per symbol spanning the
    entire requested price window. Its available_time must be <= decision_time.
    Every action on an ex-date in the price window must be verified and known by
    the cutoff. A later-retrieved action is not silently filtered out.

    cash_per_share is pre-tax cash per old share; share_factor is new shares per
    old share. Complex/rights actions are rejected. This wealth-index output is
    for cross-session return/structure features, never execution prices.
    """
    cutoff = _utc(decision_time, "decision_time")
    missing = PRICE_REQUIRED - set(prices.columns)
    if missing:
        raise ValueError("price schema incomplete: " + ",".join(sorted(missing)))
    if ACTION_REQUIRED - set(actions.columns):
        raise ValueError("action schema incomplete: " + ",".join(sorted(ACTION_REQUIRED - set(actions.columns))))
    if COVERAGE_REQUIRED - set(coverage.columns):
        raise ValueError("coverage schema incomplete: " + ",".join(sorted(COVERAGE_REQUIRED - set(coverage.columns))))

    requested = sorted({str(s) for s in (symbols if symbols is not None else prices.symbol.unique())})
    outputs = []
    for symbol in requested:
        p = prices[prices.symbol.astype(str) == symbol].copy()
        if p.empty:
            continue
        try:
            p["trade_date"] = pd.to_datetime(p["trade_date"], errors="coerce").dt.normalize()
            if p["trade_date"].isna().any() or p["trade_date"].duplicated().any():
                outputs.append(_blocked(p, "INVALID_PRICE_DATES", "invalid_or_duplicate_trade_date").prices)
                continue
            if (p[["source_id", "source_revision"]].isna().any().any()
                    or p["source_id"].astype(str).str.strip().eq("").any()
                    or p["source_revision"].astype(str).str.strip().eq("").any()):
                outputs.append(_blocked(p, "INVALID_PRICE_LINEAGE", "price_source_lineage_required").prices)
                continue
            p["available_time"] = _timestamps(p, "available_time", "prices")
            if (p["available_time"] > cutoff).any():
                outputs.append(_blocked(p, "PRICE_NOT_AVAILABLE_AT_DECISION", "price_observation_after_cutoff").prices)
                continue
            numeric = p[["open", "high", "low", "close"]].apply(pd.to_numeric, errors="coerce")
            values = numeric.to_numpy(dtype=float)
            if (not np.isfinite(values).all() or (numeric <= 0).any().any()
                    or (numeric.high < numeric[["open", "low", "close"]].max(axis=1)).any()
                    or (numeric.low > numeric[["open", "high", "close"]].min(axis=1)).any()):
                outputs.append(_blocked(p, "INVALID_PRICE_VALUES", "invalid_ohlc").prices)
                continue
            p[["open", "high", "low", "close"]] = numeric
            p = p.sort_values("trade_date").reset_index(drop=True)
            first_date, last_date = p.trade_date.iloc[0], p.trade_date.iloc[-1]

            c = coverage[coverage.symbol.astype(str) == symbol].copy()
            if len(c) != 1:
                outputs.append(_blocked(p, "MISSING_OR_DUPLICATE_COVERAGE", "exactly_one_coverage_attestation_required").prices)
                continue
            att = c.iloc[0]
            try:
                att_time = _utc(att.available_time, "coverage.available_time")
                cov_start = pd.Timestamp(att.coverage_start).normalize()
                cov_end = pd.Timestamp(att.coverage_end).normalize()
            except Exception:
                outputs.append(_blocked(p, "INVALID_COVERAGE_ATTESTATION", "invalid_coverage_metadata").prices)
                continue
            if (str(att.coverage_status) != "COMPLETE" or pd.isna(cov_start) or pd.isna(cov_end)
                    or cov_start > first_date or cov_end < last_date or att_time > cutoff
                    or pd.isna(att.source_id) or pd.isna(att.source_revision)
                    or not str(att.source_id).strip() or not str(att.source_revision).strip()):
                outputs.append(_blocked(p, "COVERAGE_NOT_VERIFIED", "coverage_must_be_complete_cover_window_and_known_by_cutoff").prices)
                continue

            a = actions[actions.symbol.astype(str) == symbol].copy()
            if not a.empty:
                a["ex_date"] = pd.to_datetime(a.ex_date, errors="coerce").dt.normalize()
                if a.ex_date.isna().any() or a.duplicated(["ex_date", "action_type"]).any():
                    outputs.append(_blocked(p, "INVALID_ACTION_LEDGER", "invalid_or_duplicate_action_key").prices)
                    continue
                # Only events inside the covered price window affect this series.
                a = a[(a.ex_date >= first_date) & (a.ex_date <= last_date)].copy()
                if not a.empty:
                    a["available_time"] = _timestamps(a, "available_time", "actions")
                    if (a[["source_id", "source_revision"]].isna().any().any()
                            or a["source_id"].astype(str).str.strip().eq("").any()
                            or a["source_revision"].astype(str).str.strip().eq("").any()):
                        outputs.append(_blocked(p, "INVALID_ACTION_LINEAGE", "action_source_lineage_required").prices)
                        continue
                    if (a.available_time > cutoff).any():
                        outputs.append(_blocked(p, "ACTION_NOT_AVAILABLE_AT_DECISION", "relevant_action_retrieved_after_cutoff").prices)
                        continue
                    if (a.verified_status.astype(str) != "VERIFIED").any():
                        outputs.append(_blocked(p, "UNVERIFIED_CORPORATE_ACTIONS", "action_status_not_verified").prices)
                        continue
                    if not a.action_type.astype(str).isin(SUPPORTED_ACTIONS).all():
                        outputs.append(_blocked(p, "UNSUPPORTED_ACTION", "rights_or_complex_action_not_supported").prices)
                        continue
                    nums = a[["cash_per_share", "share_factor"]].apply(pd.to_numeric, errors="coerce")
                    if (not np.isfinite(nums.to_numpy(dtype=float)).all()
                            or (nums.cash_per_share < 0).any() or (nums.share_factor <= 0).any()):
                        outputs.append(_blocked(p, "INVALID_ACTION_LEDGER", "invalid_cash_or_share_factor").prices)
                        continue
                    a[["cash_per_share", "share_factor"]] = nums
                    if first_date in set(a.ex_date):
                        outputs.append(_blocked(p, "ACTION_BEFORE_PRICE_COVERAGE", "first_price_date_action_has_no_prior_close").prices)
                        continue
            else:
                a["ex_date"] = pd.Series(dtype="datetime64[ns]")

            by_date = {}
            for ex_date, group in a.groupby("ex_date"):
                by_date[pd.Timestamp(ex_date)] = (
                    float(group.cash_per_share.sum()),
                    float(group.share_factor.prod()),
                    group.available_time.max(),
                )

            close_index = [100.0]
            high_index = [100.0 * float(p.high.iloc[0]) / float(p.close.iloc[0])]
            availability = [max(att_time, p.available_time.iloc[0])]
            prev_close = float(p.close.iloc[0])
            cumulative_available = availability[0]
            for row in p.iloc[1:].itertuples(index=False):
                cash, shares, event_available = by_date.get(
                    pd.Timestamp(row.trade_date), (0.0, 1.0, pd.NaT)
                )
                wealth_close = cash + shares * float(row.close)
                wealth_high = cash + shares * float(row.high)
                if prev_close <= 0 or wealth_close <= 0 or wealth_high <= 0:
                    raise ValueError("non_positive_total_return_wealth")
                close_index.append(close_index[-1] * wealth_close / prev_close)
                high_index.append(close_index[-1] * wealth_high / prev_close)
                cumulative_available = max(cumulative_available, row.available_time)
                if not pd.isna(event_available):
                    cumulative_available = max(cumulative_available, event_available)
                availability.append(cumulative_available)
                prev_close = float(row.close)

            p["pit_total_return_close"] = close_index
            p["pit_total_return_high"] = high_index
            p["pit_total_return_available_time"] = availability
            p["corporate_action_status"] = "VERIFIED"
            p["pit_total_return_reason"] = "complete_time_stamped_coverage_and_verified_actions"
            outputs.append(p)
        except (TypeError, ValueError, KeyError) as exc:
            outputs.append(_blocked(p, "PIT_BUILD_REJECTED", str(exc)).prices)

    return pd.concat(outputs, ignore_index=True) if outputs else pd.DataFrame()
