"""Forward-only event price-in observation; never a standalone trade signal.

The event clock is first_seen_at (when the model could first use the candidate),
not an unverified publisher timestamp. Historical bars before that date measure
pre-discovery run-up; only later exchange sessions measure post-discovery reaction.
All outputs remain diagnostic and are explicitly ineligible for PIT backtests or
formal stock scores.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

_REQUIRED = {
    "trade_date", "industry_name", "industry_return_1d",
    "market_return_1d", "turnover", "market_turnover",
    "observation_available_time",
}


def _compound(values: pd.Series) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty or not np.isfinite(clean.to_numpy(dtype=float)).all():
        return float("nan")
    return float((1.0 + clean).prod() - 1.0)


def evaluate_event_price_in(
    events: list[dict], industry_history: pd.DataFrame, *, decision_time,
) -> tuple[pd.DataFrame, dict]:
    """Measure pre-discovery run-up and post-discovery sector excess returns.

    industry_history must be a retrieval-time panel with one row per
    industry/session and an observation_available_time for the current fetch.
    A row is post-event only when its exchange session is later than the local
    first_seen_at date and its observation was available by decision_time.
    """
    cutoff = pd.Timestamp(decision_time)
    if cutoff.tzinfo is None:
        raise ValueError("decision_time must be timezone-aware")
    missing = _REQUIRED - set(industry_history.columns)
    if missing:
        raise ValueError("industry_history missing: " + ",".join(sorted(missing)))
    history = industry_history.copy()
    history["trade_date"] = pd.to_datetime(history["trade_date"], errors="coerce").dt.normalize()
    history["observation_available_time"] = pd.to_datetime(
        history["observation_available_time"], errors="coerce", utc=True
    )
    for field in ("industry_return_1d", "market_return_1d", "turnover", "market_turnover"):
        history[field] = pd.to_numeric(history[field], errors="coerce")
    cutoff_utc = cutoff.tz_convert("UTC")
    history = history.loc[
        history["trade_date"].notna()
        & history["observation_available_time"].notna()
        & (history["observation_available_time"] <= cutoff_utc)
    ].copy()
    history["excess_return_1d"] = (
        (1.0 + history["industry_return_1d"])
        / (1.0 + history["market_return_1d"].replace(-1.0, np.nan))
    ) - 1.0
    history["turnover_share"] = (
        history["turnover"] / history["market_turnover"].where(history["market_turnover"] > 0)
    )
    rows = []
    for event in events or []:
        if not isinstance(event, dict):
            continue
        event_id = str(event.get("candidate_id") or event.get("event_id") or "").strip()
        first_seen = pd.to_datetime(event.get("first_seen_at"), utc=True, errors="coerce")
        industries = event.get("mapped_industries") or event.get("industry_mapping") or []
        if isinstance(industries, str):
            industries = [industries]
        industries = [str(x).strip() for x in industries if str(x).strip()]
        if not event_id or pd.isna(first_seen) or not industries:
            rows.append({
                "event_id": event_id or "UNKNOWN",
                "price_in_state": "UNKNOWN",
                "reason": "missing_event_clock_or_industry_mapping",
            })
            continue
        event_date = first_seen.tz_convert("Asia/Shanghai").tz_localize(None).normalize()
        subset = history.loc[history["industry_name"].astype(str).isin(industries)].copy()
        if subset.empty:
            rows.append({
                "event_id": event_id, "first_seen_at": first_seen.isoformat(),
                "mapped_industries": industries, "price_in_state": "UNKNOWN",
                "reason": "no_aligned_industry_history",
            })
            continue
        daily = subset.groupby("trade_date", as_index=False).agg(
            excess_return_1d=("excess_return_1d", "mean"),
            turnover_share=("turnover_share", "mean"),
        ).sort_values("trade_date")
        pre = daily.loc[daily["trade_date"] <= event_date].tail(20)
        post = daily.loc[daily["trade_date"] > event_date]
        pre5 = _compound(pre["excess_return_1d"].tail(5))
        pre20 = _compound(pre["excess_return_1d"].tail(20))
        post1_rows, post3_rows, post5_rows = post.head(1), post.head(3), post.head(5)
        post1 = _compound(post1_rows["excess_return_1d"])
        post3 = _compound(post3_rows["excess_return_1d"])
        post5 = _compound(post5_rows["excess_return_1d"])
        pre_share = pre["turnover_share"].tail(5).mean()
        post_share = post.head(3)["turnover_share"].mean()
        share_ratio = (
            float(post_share / pre_share)
            if pd.notna(pre_share) and pre_share > 0 and pd.notna(post_share)
            else float("nan")
        )
        post_count = int(len(post))
        if len(pre) < 5 or not np.isfinite(pre20):
            state, reason = "UNKNOWN", "insufficient_pre_event_sessions"
        elif np.isfinite(post3) and post3 <= -0.03 and np.isfinite(post1) and post1 > 0:
            state, reason = "POST_EVENT_FADE", "initial_positive_reaction_reversed"
        elif (
            np.isfinite(pre20) and pre20 >= 0.08
            and (not np.isfinite(post1) or post1 < 0.01)
        ):
            state, reason = "FULLY_PRICED", "large_pre_discovery_excess_runup_without_new_repricing"
        elif np.isfinite(post1) and post1 >= 0.02 and np.isfinite(share_ratio) and share_ratio >= 1.2:
            state, reason = "PRICING_IN", "positive_post_event_excess_with_turnover_share_expansion"
        elif np.isfinite(pre5) and pre5 >= 0.03 and (not np.isfinite(post1) or post1 > 0):
            state, reason = "PARTIALLY_PRICED", "pre_discovery_runup_with_nonnegative_follow_through"
        elif post_count == 0 or (np.isfinite(post1) and post1 < 0.01):
            state, reason = "UNPRICED", "limited_pre_discovery_runup_and_no_material_positive_reaction"
        else:
            state, reason = "PARTIALLY_PRICED", "early_or_mixed_post_event_reaction"
        direction = (
            "positive" if np.isfinite(post1) and post1 > 0.005
            else "negative" if np.isfinite(post1) and post1 < -0.005
            else "flat" if np.isfinite(post1) else "not_observed"
        )
        rows.append({
            "event_id": event_id,
            "first_seen_at": first_seen.isoformat(),
            "event_session_date": event_date.date().isoformat(),
            "mapped_industries": industries,
            "pre_discovery_excess_5d": pre5,
            "pre_discovery_excess_20d": pre20,
            "post_discovery_excess_1d": post1,
            "post_discovery_excess_3d": post3,
            "post_discovery_excess_5d": post5,
            "post_discovery_sessions_observed": post_count,
            "turnover_share_ratio_post3_vs_pre5": share_ratio,
            "price_reaction_direction": direction,
            "price_in_state": state,
            "reason": reason,
            "impact_direction_inferred": False,
            "historical_pit_eligible": False,
            "formal_stock_score_eligible": False,
            "basis": "first_seen_forward_observation_and_retrieval_time_industry_history",
        })
    result = pd.DataFrame(rows)
    audit = {
        "status": "READY_DIAGNOSTIC_ONLY" if not result.empty else "NO_EVENT_CANDIDATES",
        "event_count": int(len(result)),
        "state_counts": result["price_in_state"].value_counts().to_dict() if not result.empty else {},
        "historical_pit_eligible": False,
        "formal_stock_score_eligible": False,
        "impact_direction_inferred": False,
    }
    return result, audit
