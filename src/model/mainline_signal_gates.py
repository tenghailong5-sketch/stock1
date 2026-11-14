"""Independent evidence gates for mainline ignition and lifecycle research.

No threshold search is performed here. Missing observations remain nullable;
an incomplete policy archive cannot be interpreted as a negative catalyst.
"""
from __future__ import annotations

from bisect import bisect_left

import pandas as pd


EMOTION_FIELDS = (
    "breadth", "limit_up_count", "limit_down_count",
    "highest_observed_limit_up_streak", "strong_pool_count",
    "board_failure_rate", "promotion_rate", "opened_rate",
)


def audit_emotion_history(frame: pd.DataFrame) -> dict:
    """Return field-level availability audit without imputing missing values."""
    if "trade_date" not in frame.columns:
        raise ValueError("emotion history requires trade_date")
    total = int(len(frame))
    fields = {}
    for field in EMOTION_FIELDS:
        present = (
            int(pd.to_numeric(frame[field], errors="coerce").notna().sum())
            if field in frame.columns else 0
        )
        fields[field] = {
            "present": present,
            "missing": total - present,
            "coverage": float(present / total) if total else None,
            "status": (
                "MISSING_FIELD" if field not in frame.columns
                else "READY" if total and present == total
                else "PARTIAL" if present else "INSUFFICIENT"
            ),
        }
    return {
        "sessions": total,
        "fields": fields,
        "missing_values_imputed": 0,
        "zero_filled": 0,
        "historical_pit_eligible": False,
    }


def attach_startup_evidence(
    industry: pd.DataFrame,
    emotion: pd.DataFrame,
    policy_events: pd.DataFrame,
    *,
    policy_archive_complete: bool = False,
    policy_window_sessions: int = 5,
) -> pd.DataFrame:
    """Attach three separately auditable ignition components to industry rows.

    A startup is CONFIRMED only when an ICE-to-RECOVERY reversal, a visible
    positive mapped policy catalyst, and rising industry turnover concentration
    all coexist. If the policy corpus is not certified complete, absence of a
    matched positive event remains UNKNOWN, never False.
    """
    required_industry = {"trade_date", "industry_code", "turnover_share",
                         "turnover_share_change_3d", "strength_3d_percentile",
                         "relative_strength_change_2d"}
    missing = required_industry - set(industry.columns)
    if missing:
        raise ValueError("missing startup industry fields: " + ",".join(sorted(missing)))
    required_emotion = {"trade_date", "market_cycle"}
    missing = required_emotion - set(emotion.columns)
    if missing:
        raise ValueError("missing startup emotion fields: " + ",".join(sorted(missing)))
    required_policy = {"available_at", "industry_codes", "impact_direction"}
    missing = required_policy - set(policy_events.columns)
    if missing:
        raise ValueError("missing startup policy fields: " + ",".join(sorted(missing)))
    if policy_window_sessions < 1:
        raise ValueError("policy_window_sessions must be positive")

    out = industry.copy()
    out["trade_date"] = pd.to_datetime(out.trade_date, errors="coerce").dt.normalize()
    if out.trade_date.isna().any():
        raise ValueError("invalid industry trade_date")
    emo = emotion.copy()
    emo["trade_date"] = pd.to_datetime(emo.trade_date, errors="coerce").dt.normalize()
    if emo.trade_date.isna().any() or emo.trade_date.duplicated().any():
        raise ValueError("emotion history must have unique valid trade dates")
    emo = emo.sort_values("trade_date").reset_index(drop=True)
    prior_ice = emo.market_cycle.eq("ICE").shift(1, fill_value=False)
    recent_ice = prior_ice.copy()
    for lag in range(2, policy_window_sessions + 1):
        recent_ice |= emo.market_cycle.eq("ICE").shift(lag, fill_value=False)
    recovery_after_ice = emo.market_cycle.eq("RECOVERY") & recent_ice
    emo["ice_reversal"] = recovery_after_ice
    emo_map = emo.set_index("trade_date")["ice_reversal"].to_dict()
    out["emotion_reversal_component"] = out.trade_date.map(emo_map).astype("boolean")

    events = policy_events.copy()
    events["available_at"] = pd.to_datetime(events.available_at, errors="coerce", utc=True)
    if events.available_at.isna().any():
        raise ValueError("policy available_at contains invalid timestamps")
    if not events.impact_direction.isin({"positive", "negative", "mixed", "unclear"}).all():
        raise ValueError("policy impact_direction uses an unknown value")
    if events.industry_codes.map(lambda x: not isinstance(x, (list, tuple, set))).any():
        raise ValueError("policy industry_codes must be explicit collections")
    # Only explicitly positive, source-backed policy events can support an
    # ignition. Mixed/negative/unclear events are not positive startup evidence.
    events = events[events.impact_direction.eq("positive")].copy()
    dates = sorted(pd.Timestamp(date) for date in out.trade_date.unique())
    date_position = {date: i for i, date in enumerate(dates)}
    # A policy event is usable at the first trading-session close (15:00
    # Shanghai time) at or after its actual availability timestamp. This
    # moves weekend/holiday and post-close publications forward, preventing
    # both missed events and same-day look-ahead.
    session_close_cutoffs = [
        date.tz_localize("Asia/Shanghai") + pd.Timedelta(hours=15)
        for date in dates
    ]
    event_sessions = []
    for event in events.itertuples(index=False):
        event_pos = bisect_left(session_close_cutoffs, event.available_at)
        if event_pos < len(dates):
            event_sessions.append((
                event_pos,
                {str(code) for code in event.industry_codes},
            ))
    matched = []
    for row in out.itertuples(index=False):
        pos = date_position[pd.Timestamp(row.trade_date)]
        first = max(0, pos - policy_window_sessions + 1)
        industry_code = str(row.industry_code)
        matched.append(any(
            first <= event_pos <= pos and industry_code in event_codes
            for event_pos, event_codes in event_sessions
        ))
    policy_values = pd.Series(matched, index=out.index, dtype="boolean")
    # A positively matched, source-backed event is known even when the archive
    # is incomplete. Only absence is UNKNOWN until archive completeness is
    # certified; otherwise valid positive catalysts are discarded.
    if not policy_archive_complete:
        policy_values = policy_values.mask(~policy_values, pd.NA)
    out["policy_catalyst_component"] = policy_values

    numeric = ["turnover_share", "turnover_share_change_3d",
               "strength_3d_percentile", "relative_strength_change_2d"]
    for field in numeric:
        out[field] = pd.to_numeric(out[field], errors="coerce")
    # Preserve each capital-flow condition separately so a failed composite
    # gate can be diagnosed without relaxing the frozen confirmation rule.
    capital_checks = {
        "capital_turnover_share_positive": ("turnover_share", lambda x: x > 0),
        "capital_turnover_share_increasing": ("turnover_share_change_3d", lambda x: x > 0),
        "capital_leadership_top_quintile": ("strength_3d_percentile", lambda x: x >= 0.80),
        "capital_relative_strength_accelerating": ("relative_strength_change_2d", lambda x: x > 0),
    }
    for output_field, (input_field, predicate) in capital_checks.items():
        values = pd.Series(pd.NA, index=out.index, dtype="boolean")
        known = out[input_field].notna()
        values.loc[known] = predicate(out.loc[known, input_field])
        out[output_field] = values

    flow_known = out[numeric].notna().all(axis=1)
    out["capital_concentration_component"] = pd.Series(
        pd.NA, index=out.index, dtype="boolean"
    )
    out.loc[flow_known, "capital_concentration_component"] = out.loc[
        flow_known, list(capital_checks)
    ].all(axis=1)
    components = [
        "emotion_reversal_component", "policy_catalyst_component",
        "capital_concentration_component",
    ]
    out["startup_evidence_status"] = "UNKNOWN"
    all_known = out[components].notna().all(axis=1)
    out.loc[all_known, "startup_evidence_status"] = (
        out.loc[all_known, components].all(axis=1).map(
            {True: "CONFIRMED", False: "NOT_CONFIRMED"}
        )
    )
    out["startup_live_eligible"] = False
    out["historical_pit_eligible"] = False
    return out


def evaluate_peak_and_decay_independently(frame: pd.DataFrame) -> pd.DataFrame:
    """Emit separate peak and decay evidence; neither label implies the other."""
    required = {"strength_percentile", "turnover_share_percentile",
                "turnover_share_change_3d", "relative_strength_change_3d",
                "relative_strength_5d"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError("missing lifecycle fields: " + ",".join(sorted(missing)))
    out = frame.copy()
    fields = list(required)
    for field in fields:
        out[field] = pd.to_numeric(out[field], errors="coerce")
    known = out[fields].notna().all(axis=1)
    out["peak_evidence"] = pd.Series(pd.NA, index=out.index, dtype="boolean")
    out["decay_evidence"] = pd.Series(pd.NA, index=out.index, dtype="boolean")
    out.loc[known, "peak_evidence"] = (
        (out.loc[known, "strength_percentile"] >= 0.80)
        & (out.loc[known, "turnover_share_percentile"] >= 0.65)
        & (out.loc[known, "relative_strength_5d"] > 0)
        & (out.loc[known, "turnover_share_change_3d"] <= 0)
        & (out.loc[known, "relative_strength_change_3d"] <= 0)
    )
    out.loc[known, "decay_evidence"] = (
        (out.loc[known, "strength_percentile"] < 0.50)
        & (out.loc[known, "turnover_share_percentile"] < 0.50)
        & (out.loc[known, "relative_strength_5d"] < 0)
        & (out.loc[known, "turnover_share_change_3d"] < 0)
        & (out.loc[known, "relative_strength_change_3d"] < 0)
    )
    out["lifecycle_data_complete"] = known
    return out


def gate_ignition_labels(frame: pd.DataFrame) -> pd.DataFrame:
    """Demote unverified ignition candidates; only three known-positive gates confirm."""
    required = {"mainline_phase", "startup_evidence_status"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError("missing ignition gate fields: " + ",".join(sorted(missing)))
    out = frame.copy()
    out["mainline_phase_candidate"] = out["mainline_phase"]
    out["mainline_phase_gated"] = out["mainline_phase_candidate"]
    candidate = out["mainline_phase_candidate"].eq("IGNITION")
    confirmed = out["startup_evidence_status"].eq("CONFIRMED").fillna(False)
    out.loc[candidate & ~confirmed, "mainline_phase_gated"] = "IGNITION_UNCONFIRMED"
    out["startup_live_eligible"] = False
    return out
