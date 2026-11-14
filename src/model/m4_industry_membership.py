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


STOCK_BAR_REQUIRED_COLUMNS = {
    "symbol", "trade_date", "amount", "available_time",
    "historical_pit_eligible", "availability_basis",
}


def aggregate_pit_industry_turnover(
    stock_bars: pd.DataFrame,
    membership: pd.DataFrame,
    expected_symbols,
    *,
    trade_date,
    decision_time,
    scope_id: str,
) -> tuple[pd.DataFrame, dict]:
    """Aggregate stock turnover by industry only from complete, PIT-safe inputs.

    Every expected security must have exactly one eligible daily bar and one
    PIT-valid industry mapping. Retrieval-time historical bars are excluded,
    even when their trade_date predates the decision timestamp.
    """
    symbols = [str(value).strip() for value in expected_symbols]
    if not symbols or any(not value for value in symbols):
        raise ValueError("expected_symbols must contain non-empty symbols")
    if len(symbols) != len(set(symbols)):
        raise ValueError("expected_symbols contains duplicates")
    if stock_bars is None or not isinstance(stock_bars, pd.DataFrame):
        raise ValueError("stock_bars must be a DataFrame")
    missing = STOCK_BAR_REQUIRED_COLUMNS - set(stock_bars.columns)
    if missing:
        raise ValueError("stock_bars missing required columns: " + ",".join(sorted(missing)))

    cutoff = pd.Timestamp(decision_time)
    if cutoff.tzinfo is None:
        raise ValueError("decision_time must be timezone-aware")
    target_date = pd.Timestamp(trade_date).normalize()
    if pd.isna(target_date):
        raise ValueError("trade_date must be valid")
    if target_date.date() > cutoff.tz_convert("Asia/Shanghai").date():
        raise ValueError("trade_date is after decision_time")

    bars = stock_bars.copy()
    bars["symbol"] = bars["symbol"].astype(str).str.strip()
    dates = pd.to_datetime(bars["trade_date"], errors="coerce").dt.normalize()
    available = pd.to_datetime(bars["available_time"], errors="coerce")
    if dates.isna().any():
        raise ValueError("stock bar trade_date must be valid")
    if available.isna().any() or available.dt.tz is None:
        raise ValueError("stock bar available_time must be valid and timezone-aware")
    bars["trade_date"] = dates
    bars["available_time"] = available.dt.tz_convert("UTC")
    bars["amount"] = pd.to_numeric(bars["amount"], errors="coerce")
    if bars["amount"].isna().any() or (~bars["amount"].map(pd.notna)).any():
        raise ValueError("stock bar amount must be numeric")
    if (bars["amount"] < 0).any():
        raise ValueError("stock bar amount must be non-negative")

    target = bars.loc[
        bars["symbol"].isin(symbols) & (bars["trade_date"] == target_date)
    ].copy()
    known = target.loc[target["available_time"] <= cutoff.tz_convert("UTC")].copy()
    eligible = known.loc[
        known["historical_pit_eligible"].map(lambda value: value is True or isinstance(value, bool) and value)
        & known["availability_basis"].astype(str).str.strip().eq("source_published_at")
    ].copy()
    eligible_count = int(eligible["symbol"].nunique())
    audit = {
        "scope_id": scope_id,
        "trade_date": target_date.date().isoformat(),
        "decision_time": cutoff.isoformat(),
        "expected_universe_size": len(symbols),
        "stock_bar_coverage": eligible_count / len(symbols),
        "stock_bar_eligible_symbols": eligible_count,
        "stock_bar_retrieval_only_symbols": int(
            known.loc[known["historical_pit_eligible"].map(lambda value: value is False or isinstance(value, bool) and not value), "symbol"].nunique()
        ),
        "historical_pit_eligible": False,
        "status": "incomplete_pit_stock_bars",
    }
    if eligible["symbol"].duplicated().any():
        audit["status"] = "duplicate_pit_stock_bars"
        return pd.DataFrame(columns=["industry_code", "stock_count", "turnover"]), audit
    if set(eligible["symbol"]) != set(symbols):
        return pd.DataFrame(columns=["industry_code", "stock_count", "turnover"]), audit

    selected_membership, membership_audit = select_pit_industry_membership(
        membership, symbols, trade_date=target_date,
        decision_time=cutoff, scope_id=scope_id,
    )
    audit["membership_status"] = membership_audit["status"]
    audit["membership_coverage"] = membership_audit["membership_coverage"]
    if membership_audit["status"] != "ready":
        audit["status"] = "incomplete_pit_membership"
        return pd.DataFrame(columns=["industry_code", "stock_count", "turnover"]), audit

    joined = eligible[["symbol", "amount"]].merge(
        selected_membership[["symbol", "industry_code"]],
        on="symbol", how="left", validate="one_to_one",
    )
    if joined["industry_code"].isna().any():
        audit["status"] = "unmapped_pit_stock_bars"
        return pd.DataFrame(columns=["industry_code", "stock_count", "turnover"]), audit

    result = (
        joined.groupby("industry_code", as_index=False)
        .agg(stock_count=("symbol", "nunique"), turnover=("amount", "sum"))
        .sort_values("industry_code")
        .reset_index(drop=True)
    )
    audit["status"] = "ready"
    audit["historical_pit_eligible"] = True
    audit["industry_count"] = int(len(result))
    return result, audit
