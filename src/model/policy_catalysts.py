"""Point-in-time policy/event contract. No inferred events or invented sentiment scores."""
from __future__ import annotations

import pandas as pd


REQUIRED = {"event_id", "published_at", "available_at", "source_url",
            "source_id", "title", "catalyst_type", "industry_codes",
            "impact_direction", "confidence", "confidence_basis", "availability_basis"}


def visible_policy_events(events: pd.DataFrame, *, decision_time) -> tuple[pd.DataFrame, dict]:
    """Return only source-backed events actually available by the decision cutoff."""
    missing = REQUIRED - set(events.columns)
    if missing:
        raise ValueError("missing policy-event fields: " + ",".join(sorted(missing)))
    cutoff = pd.Timestamp(decision_time)
    if cutoff.tzinfo is None:
        raise ValueError("decision_time must be timezone-aware")
    out = events.copy()
    for col in ("published_at", "available_at"):
        out[col] = pd.to_datetime(out[col], errors="coerce", utc=True)
    invalid = out[list(REQUIRED - {"industry_codes"})].isna().any(axis=1)
    invalid |= out.source_url.astype(str).str.match(r"^https?://") == False
    invalid |= out.industry_codes.map(lambda x: not isinstance(x, (list, tuple, set)) or len(x) == 0)
    if invalid.any():
        raise ValueError("invalid policy-event records; source and industry mapping are mandatory")
    if (out.published_at > out.available_at).any():
        raise ValueError("event cannot be available before publication")
    if not out.impact_direction.isin({"positive", "negative", "mixed", "unclear"}).all():
        raise ValueError("impact_direction must use the controlled event vocabulary")
    confidence = pd.to_numeric(out.confidence, errors="coerce")
    if confidence.isna().any() or ((confidence < 0) | (confidence > 1)).any():
        raise ValueError("confidence must be a numeric value within [0,1]")
    if out.confidence_basis.astype(str).str.strip().eq("").any():
        raise ValueError("confidence_basis is required for auditability")
    if not out.availability_basis.isin({
        "publisher_timestamp", "publisher_date_eod_conservative",
        "document_date_plus_publication_date",
    }).all():
        raise ValueError("availability_basis must describe the source timestamp precision")
    out["confidence"] = confidence
    cutoff = cutoff.tz_convert("UTC")
    visible = out.loc[(out.published_at <= cutoff) & (out.available_at <= cutoff)].copy()
    visible["historical_pit_eligible"] = False
    audit = {
        "input_rows": int(len(out)),
        "visible_rows": int(len(visible)),
        "future_or_not_yet_available_rows": int(len(out) - len(visible)),
        "historical_pit_eligible": False,
        "status": "READY_FORWARD_ONLY" if len(visible) else "NO_VISIBLE_EVENTS",
    }
    return visible.sort_values(["available_at", "event_id"]), audit
