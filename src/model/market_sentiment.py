"""Descriptive market-sentiment observations from provider pool snapshots.

These metrics are observational only: they are not PIT-eligible, do not infer
intraday events, and must not be fed into FHM-100 scoring without validation.
"""
from __future__ import annotations

import math
from numbers import Real

import pandas as pd

REQUIRED_POOLS = {
    "limit_up",
    "limit_down",
    "limit_up_opened",
    "previous_limit_up",
    "strong",
}


def summarize_limit_pool_sentiment(snapshots: pd.DataFrame) -> dict:
    """Summarize pool sizes and observed limit-up ladder height without inventing rates."""
    if not isinstance(snapshots, pd.DataFrame) or snapshots.empty:
        raise ValueError("limit pool snapshots are required")
    if not {"event_type", "provider_row_count", "records"}.issubset(snapshots.columns):
        raise ValueError("limit pool snapshot schema is incomplete")

    by_type = {}
    for _, row in snapshots.iterrows():
        event_type = row["event_type"]
        if event_type in by_type:
            raise ValueError(f"duplicate pool snapshot: {event_type}")
        count = row["provider_row_count"]
        if isinstance(count, bool) or not isinstance(count, Real) or not math.isfinite(float(count)) or int(count) != count or count < 0:
            raise ValueError(f"invalid provider_row_count for {event_type}")
        records = row["records"]
        if not isinstance(records, list) or len(records) != int(count):
            raise ValueError(f"records/count mismatch for {event_type}")
        by_type[event_type] = (int(count), records)

    missing = REQUIRED_POOLS.difference(by_type)
    if missing:
        raise ValueError(f"missing required pool snapshots: {sorted(missing)}")

    limit_up_records = by_type["limit_up"][1]
    streak_values = []
    for record in limit_up_records:
        if not isinstance(record, dict):
            continue
        value = record.get("连板数")
        if isinstance(value, bool):
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(numeric) and numeric >= 1 and numeric.is_integer():
            streak_values.append(int(numeric))

    return {
        "metric_scope": "descriptive_provider_pool_observation_only",
        "historical_pit_eligible": False,
        "scoring_eligible": False,
        "limit_up_count": by_type["limit_up"][0],
        "limit_down_count": by_type["limit_down"][0],
        "limit_up_opened_pool_count": by_type["limit_up_opened"][0],
        "previous_limit_up_count": by_type["previous_limit_up"][0],
        "strong_pool_count": by_type["strong"][0],
        "limit_up_streak_observed_count": len(streak_values),
        "limit_up_streak_missing_count": len(limit_up_records) - len(streak_values),
        "highest_observed_limit_up_streak": max(streak_values) if streak_values else None,
        "opened_rate": None,
        "opened_rate_status": "not_computed_provider_pool_semantics_not_validated",
    }
