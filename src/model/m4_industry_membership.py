"""Point-in-time historical industry-membership selection.

Membership is usable only when its validity interval covers the target trade
date AND the exact record was available by the decision timestamp. Snapshot
observation dates are not silently promoted into historical validity intervals.
"""
from __future__ import annotations

import hashlib

import pandas as pd


REQUIRED_COLUMNS = {
    "symbol", "industry_code", "valid_from", "valid_to",
    "available_time", "source_id", "source_revision", "snapshot_scope",
}


def select_pit_industry_membership(
    membership: pd.DataFrame,
    expected_symbols,
    *,
    trade_date,
    decision_time,
    scope_id: str,
) -> tuple[pd.DataFrame, dict]:
    """Select one historically valid industry per expected symbol, fail-closed.

    valid_from is inclusive and valid_to is exclusive; null valid_to means
    open-ended membership. available_time must be the actual first time this
    exact source revision was available to the system. The function does not
    infer validity from observed_at or retrieval time.

    If any expected symbol lacks a qualifying mapping, or if the latest
    available records conflict, no membership rows are returned.
    """
    if not isinstance(scope_id, str) or not scope_id.strip():
        raise ValueError("scope_id must be a non-empty string")
    if membership is None or not isinstance(membership, pd.DataFrame):
        raise ValueError("membership must be a DataFrame")
    missing = REQUIRED_COLUMNS - set(membership.columns)
    if missing:
        raise ValueError("membership missing required columns: " + ",".join(sorted(missing)))

    symbols = [str(value).strip() for value in expected_symbols]
    if not symbols or any(not value for value in symbols):
        raise ValueError("expected_symbols must contain non-empty symbols")
    if len(symbols) != len(set(symbols)):
        raise ValueError("expected_symbols contains duplicates")

    cutoff = pd.Timestamp(decision_time)
    if cutoff.tzinfo is None:
        raise ValueError("decision_time must be timezone-aware")
    target_date = pd.Timestamp(trade_date)
    if pd.isna(target_date):
        raise ValueError("trade_date must be valid")
    target_date = target_date.normalize()
    if target_date.date() > cutoff.tz_convert("Asia/Shanghai").date():
        raise ValueError("trade_date is after decision_time")

    x = membership.copy()
    available = pd.to_datetime(x["available_time"], errors="coerce")
    if available.isna().any() or available.dt.tz is None:
        raise ValueError("available_time must be valid and timezone-aware")
    x["available_time"] = available.dt.tz_convert("UTC")
    x = x.loc[x["available_time"] <= cutoff.tz_convert("UTC")].copy()
    x = x.loc[x["snapshot_scope"].astype(str).str.strip() == scope_id].copy()

    # Validate only records that could have been known at this decision.
    x["valid_from"] = pd.to_datetime(x["valid_from"], errors="coerce")
    x["valid_to"] = pd.to_datetime(x["valid_to"], errors="coerce")
    if x["valid_from"].isna().any():
        raise ValueError("valid_from must be a valid date")
    if (x["valid_to"].notna() & (x["valid_to"] <= x["valid_from"])).any():
        raise ValueError("valid_to must be later than valid_from (exclusive)")
    for column in ("symbol", "industry_code", "source_id", "source_revision", "snapshot_scope"):
        values = x[column]
        if values.isna().any() or values.astype(str).str.strip().eq("").any():
            raise ValueError(f"invalid {column}")
        x[column] = values.astype(str).str.strip()

    eligible = x.loc[
        (x["valid_from"] <= target_date)
        & (x["valid_to"].isna() | (target_date < x["valid_to"]))
        & x["symbol"].isin(symbols)
    ].copy()

    expected_hash = hashlib.sha256("\n".join(sorted(symbols)).encode("utf-8")).hexdigest()
    empty = membership.iloc[0:0].copy()

    def audit(status: str, selected_count: int = 0, selected_hash=None):
        return {
            "scope_id": scope_id,
            "trade_date": target_date.date().isoformat(),
            "decision_time": cutoff.isoformat(),
            "expected_universe_size": len(symbols),
            "selected_universe_size": selected_count,
            "membership_coverage": selected_count / len(symbols),
            "expected_symbols_sha256": expected_hash,
            "selected_symbols_sha256": selected_hash,
            "status": status,
            "historical_pit_eligible": status == "ready",
        }

    if eligible.empty:
        return empty, audit("no_eligible_membership")

    # Latest revision available by the decision time supersedes earlier
    # observations. Multiple latest rows for a symbol are ambiguous and block.
    latest_times = eligible.groupby("symbol")["available_time"].transform("max")
    latest = eligible.loc[eligible["available_time"] == latest_times].copy()
    if latest["symbol"].duplicated().any():
        return empty, audit("ambiguous_latest_membership")

    selected_symbols = set(latest["symbol"])
    if selected_symbols != set(symbols):
        selected_hash = hashlib.sha256("\n".join(sorted(selected_symbols)).encode("utf-8")).hexdigest()
        return empty, audit("incomplete_membership", len(selected_symbols), selected_hash)

    latest = latest.sort_values("symbol").reset_index(drop=True)
    selected_hash = hashlib.sha256("\n".join(latest["symbol"]).encode("utf-8")).hexdigest()
    return latest, audit("ready", len(latest), selected_hash)
