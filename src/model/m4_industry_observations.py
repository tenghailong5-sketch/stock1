"""Forward-only selection of complete, retrieval-time industry snapshots.

This is deliberately separate from historical PIT membership selection.
A selected result can support prospective observations only; it never
establishes historical validity intervals.
"""
from __future__ import annotations

import pandas as pd


REQUIRED_COLUMNS = {
    "snapshot_id", "expected_industry_count", "symbol", "industry_code", "observed_at", "available_time",
    "source_id", "source_revision", "snapshot_scope", "availability_basis",
    "historical_pit_eligible",
}


def _aware_utc_series(values: pd.Series, name: str) -> pd.Series:
    parsed = []
    for value in values:
        stamp = pd.Timestamp(value)
        if pd.isna(stamp) or stamp.tzinfo is None:
            raise ValueError(f"{name} must contain timezone-aware timestamps")
        parsed.append(stamp.tz_convert("UTC"))
    return pd.Series(parsed, index=values.index, dtype="datetime64[ns, UTC]")


def select_latest_observed_industry_snapshot(
    membership: pd.DataFrame,
    expected_symbols,
    *,
    decision_time,
    max_age="10D",
    min_symbol_coverage=0.90,
) -> tuple[pd.DataFrame, dict]:
    """Select the newest sufficiently complete visible forward-only membership batch.

    Incomplete newest batches no longer erase an older complete snapshot that
    is still fresh. Partial symbol coverage is usable for research once the
    configured floor is met; absent symbols simply retain missing industry data.
    """
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
    cutoff = cutoff.tz_convert("UTC")
    age_limit = pd.Timedelta(max_age)
    if age_limit <= pd.Timedelta(0):
        raise ValueError("max_age must be positive")
    if not 0 < float(min_symbol_coverage) <= 1:
        raise ValueError("min_symbol_coverage must be within (0,1]")

    empty = membership.iloc[0:0].copy()

    def audit(status, *, snapshot_id=None, selected=None, latest_time=None,
              skipped_batches=None, fallback_used=False):
        selected = selected if selected is not None else empty
        matched = set(selected["symbol"].astype(str)) if not selected.empty else set()
        return {
            "snapshot_id": snapshot_id,
            "decision_time": cutoff.isoformat(),
            "latest_available_time": latest_time.isoformat() if latest_time is not None else None,
            "expected_universe_size": len(symbols),
            "selected_symbol_count": len(matched),
            "membership_coverage": len(matched) / len(symbols),
            "minimum_symbol_coverage": float(min_symbol_coverage),
            "status": status,
            "fallback_used": bool(fallback_used),
            "skipped_batches": skipped_batches or [],
            "historical_pit_eligible": False,
        }

    if membership.empty:
        return empty, audit("no_snapshot")

    x = membership.copy()
    for column in ("snapshot_id", "symbol", "industry_code", "source_id", "source_revision", "snapshot_scope", "availability_basis"):
        values = x[column]
        if values.isna().any() or values.astype(str).str.strip().eq("").any():
            raise ValueError(f"invalid {column}")
        x[column] = values.astype(str).str.strip()

    if not x["snapshot_scope"].eq("current_membership_observation_only").all():
        raise ValueError("only current_membership_observation_only snapshots are accepted")
    if not x["availability_basis"].eq("retrieval_time_only").all():
        raise ValueError("industry observations must use retrieval_time_only availability")
    pit_flags = x["historical_pit_eligible"].map(
        lambda value: value is True or str(value).strip().lower() in {"true", "1", "yes"}
    )
    if x["historical_pit_eligible"].isna().any() or pit_flags.any():
        raise ValueError("historical PIT-eligible rows are not valid forward observations")

    x["available_time"] = _aware_utc_series(x["available_time"], "available_time")
    x["observed_at"] = _aware_utc_series(x["observed_at"], "observed_at")
    batch_available = x.groupby("snapshot_id")["available_time"].max()
    visible_ids = batch_available.loc[batch_available <= cutoff].sort_values(ascending=False)
    if visible_ids.empty:
        return empty, audit("no_visible_snapshot")

    skipped = []
    latest_visible_time = visible_ids.iloc[0]
    last_reason = "incomplete_universe"
    for batch_index, (snapshot_id, latest_time) in enumerate(visible_ids.items()):
        if cutoff - latest_time > age_limit:
            last_reason = "stale_snapshot"
            break
        batch = x.loc[x["snapshot_id"] == snapshot_id].copy()
        if (batch["available_time"] > cutoff).any():
            last_reason = "batch_not_fully_visible"
            skipped.append({"snapshot_id": str(snapshot_id), "reason": last_reason})
            continue
        expected_counts = pd.to_numeric(
            batch["expected_industry_count"], errors="coerce"
        ).dropna().unique()
        if len(expected_counts) != 1 or expected_counts[0] <= 0:
            last_reason = "invalid_batch_manifest"
            skipped.append({"snapshot_id": str(snapshot_id), "reason": last_reason})
            continue
        if batch["industry_code"].nunique() != int(expected_counts[0]):
            last_reason = "incomplete_industry_batch"
            skipped.append({"snapshot_id": str(snapshot_id), "reason": last_reason})
            continue
        if batch.duplicated(["symbol", "industry_code"]).any():
            last_reason = "duplicate_membership"
            skipped.append({"snapshot_id": str(snapshot_id), "reason": last_reason})
            continue

        selected = batch.loc[batch["symbol"].isin(symbols)].copy()
        matched = set(selected["symbol"].astype(str))
        coverage = len(matched) / len(symbols)
        if coverage < float(min_symbol_coverage):
            last_reason = "incomplete_universe"
            skipped.append({
                "snapshot_id": str(snapshot_id), "reason": last_reason,
                "membership_coverage": coverage,
            })
            continue

        selected = selected.sort_values(["symbol", "industry_code"]).reset_index(drop=True)
        status = "ready_forward_only" if coverage == 1.0 else "ready_forward_only_partial"
        return selected, audit(
            status, snapshot_id=str(snapshot_id), selected=selected,
            latest_time=latest_time, skipped_batches=skipped,
            fallback_used=batch_index > 0,
        )

    return empty, audit(
        last_reason, snapshot_id=str(visible_ids.index[0]),
        latest_time=latest_visible_time, skipped_batches=skipped,
    )
