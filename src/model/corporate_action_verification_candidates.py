"""Convert forward-observed corporate actions into non-authoritative verification candidates.

Each revision becomes available when that revision was actually observed, not
when the event was first seen. The adapter selects the latest revision known at
the decision cutoff and never marks an action VERIFIED.
"""
from __future__ import annotations

import pandas as pd


REQUIRED = {
    "symbol", "event_type", "event_key", "announcement_date", "ex_date",
    "cash_dividend", "bonus_share_ratio", "capitalization_ratio",
    "rights_issue_ratio", "rights_issue_price", "progress", "first_seen_at",
    "observed_at", "available_time", "source_id", "source_revision",
    "corporate_action_status",
}


def build_corporate_action_verification_candidates(
    ledger: pd.DataFrame, *, decision_time, ledger_snapshot_id: str,
) -> pd.DataFrame:
    """Return latest as-of-cutoff candidates; never VERIFIED rows."""
    cutoff = pd.Timestamp(decision_time)
    if cutoff.tzinfo is None:
        raise ValueError("decision_time must be timezone-aware")
    if not str(ledger_snapshot_id).strip():
        raise ValueError("ledger_snapshot_id is required")
    missing = REQUIRED - set(ledger.columns)
    if missing:
        raise ValueError("ledger missing columns: " + ",".join(sorted(missing)))

    x = ledger.copy()
    for col in ("first_seen_at", "observed_at", "available_time"):
        ts = pd.to_datetime(x[col], errors="coerce", utc=True)
        if ts.isna().any():
            raise ValueError(f"{col} must be valid and timezone-aware")
        x[col] = ts
    cutoff = cutoff.tz_convert("UTC")
    if (x.observed_at < x.first_seen_at).any():
        raise ValueError("revision observation cannot precede first_seen_at")
    # The source's available_time is the event's first_seen_at. For a changed
    # revision, the revision itself is not known until observed_at.
    x = x.loc[(x.first_seen_at <= cutoff) & (x.observed_at <= cutoff)].copy()
    if x.empty:
        return pd.DataFrame(columns=[
            "symbol", "event_key", "ex_date", "first_seen_at", "observed_at",
            "available_time", "source_id", "source_revision", "ledger_snapshot_id",
            "verified_status", "verification_required", "candidate_reason",
            "action_type", "cash_per_share", "share_factor",
        ])

    ambiguous = x.groupby(["event_key", "observed_at"])["source_revision"].nunique()
    if ambiguous.gt(1).any():
        raise ValueError("conflicting revisions share the same observation timestamp")
    x = x.sort_values(["event_key", "observed_at", "source_revision"])
    x = x.drop_duplicates("event_key", keep="last")

    rows = []
    for row in x.to_dict("records"):
        base = {
            "symbol": str(row["symbol"]),
            "event_key": str(row["event_key"]),
            "ex_date": row["ex_date"],
            "first_seen_at": row["first_seen_at"],
            "observed_at": row["observed_at"],
            "available_time": row["observed_at"],
            "source_id": str(row["source_id"]),
            "source_revision": str(row["source_revision"]),
            "ledger_snapshot_id": str(ledger_snapshot_id),
            "verified_status": "PENDING_VERIFICATION",
            "verification_required": True,
            "candidate_reason": None,
            "action_type": "unsupported",
            "cash_per_share": None,
            "share_factor": None,
        }
        if str(row["corporate_action_status"]) != "CAPTURED":
            base["candidate_reason"] = "source_row_not_captured"
        elif pd.isna(row["ex_date"]):
            base["candidate_reason"] = "missing_ex_date"
        elif str(row["progress"]).strip() not in {"实施", "实施完成"}:
            base["candidate_reason"] = "event_not_confirmed_implemented"
        elif row["event_type"] == "dividend_bonus":
            values = [row["cash_dividend"], row["bonus_share_ratio"], row["capitalization_ratio"]]
            if any(pd.isna(v) for v in values):
                base["candidate_reason"] = "incomplete_dividend_components"
            else:
                cash, bonus, capitalization = map(float, values)
                if min(cash, bonus, capitalization) < 0:
                    base["candidate_reason"] = "negative_dividend_component"
                else:
                    base.update({
                        "action_type": "cash_dividend" if bonus + capitalization == 0 else "bonus",
                        "cash_per_share": cash / 10.0,
                        "share_factor": 1.0 + (bonus + capitalization) / 10.0,
                        "candidate_reason": "normalized_per_old_share_pending_independent_verification",
                    })
        elif row["event_type"] == "rights_issue":
            base["candidate_reason"] = "rights_issue_requires_subscription_and_takeup_reconciliation"
        else:
            base["candidate_reason"] = "unknown_event_type"
        rows.append(base)
    return pd.DataFrame(rows)
