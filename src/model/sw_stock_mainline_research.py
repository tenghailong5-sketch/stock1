"""Research-only link between effective-dated SW stocks and live industry mainlines."""
from __future__ import annotations

import pandas as pd


def build_sw_stock_mainline_research(
    membership: pd.DataFrame,
    mainline: pd.DataFrame,
    *,
    as_of_date,
) -> tuple[pd.DataFrame, dict]:
    """Map stocks to the latest observed SW mainline without PIT/trading authority."""
    required_membership = {
        "symbol", "industry_code", "valid_from", "valid_to",
        "availability_basis", "historical_pit_eligible",
    }
    required_mainline = {"industry_code", "trade_date", "mainline_phase"}
    if membership is None or not isinstance(membership, pd.DataFrame):
        raise ValueError("membership must be a DataFrame")
    if mainline is None or not isinstance(mainline, pd.DataFrame):
        raise ValueError("mainline must be a DataFrame")
    missing = required_membership - set(membership.columns)
    if missing:
        raise ValueError("SW membership missing: " + ",".join(sorted(missing)))
    missing = required_mainline - set(mainline.columns)
    if missing:
        raise ValueError("mainline missing: " + ",".join(sorted(missing)))

    target = pd.Timestamp(as_of_date).normalize()
    if pd.isna(target):
        raise ValueError("as_of_date must be valid")
    members = membership.copy()
    members["symbol"] = members["symbol"].astype(str).str.strip().str.zfill(6)
    members["industry_code"] = members["industry_code"].astype(str).str.strip()
    members["valid_from"] = pd.to_datetime(members["valid_from"], errors="coerce").dt.normalize()
    members["valid_to"] = pd.to_datetime(members["valid_to"], errors="coerce").dt.normalize()
    if members[["valid_from"]].isna().any().any():
        raise ValueError("SW membership contains invalid valid_from")
    if not members["availability_basis"].astype(str).eq("retrieval_time_only").all():
        raise ValueError("SW membership provenance must remain retrieval_time_only")
    if members["historical_pit_eligible"].map(
        lambda value: value is True or str(value).strip().lower() in {"true", "1", "yes"}
    ).any():
        raise ValueError("PIT-eligible membership is not accepted by research mapper")

    active = members.loc[
        (members["valid_from"] <= target)
        & (members["valid_to"].isna() | (target < members["valid_to"]))
    ].copy()
    if active["symbol"].duplicated().any():
        audit = {
            "status": "AMBIGUOUS_STOCK_INDUSTRY_MAPPING",
            "as_of_date": target.date().isoformat(),
            "membership_rows": int(len(members)),
            "active_membership_rows": int(len(active)),
            "matched_symbols": 0,
            "historical_pit_eligible": False,
            "trading_authorization": False,
            "research_only": True,
        }
        return pd.DataFrame(), audit

    if mainline.empty:
        audit = {
            "status": "MAINLINE_UNAVAILABLE",
            "as_of_date": target.date().isoformat(),
            "membership_rows": int(len(active)),
            "matched_symbols": 0,
            "historical_pit_eligible": False,
            "trading_authorization": False,
            "research_only": True,
        }
        return pd.DataFrame(), audit

    panel = mainline.copy()
    panel["industry_code"] = panel["industry_code"].astype(str).str.strip()
    panel["trade_date"] = pd.to_datetime(panel["trade_date"], errors="coerce").dt.normalize()
    if panel["trade_date"].isna().any():
        raise ValueError("mainline contains invalid trade_date")
    latest_date = panel["trade_date"].max()
    latest = panel.loc[panel["trade_date"].eq(latest_date)].copy()
    if latest["industry_code"].duplicated().any():
        raise ValueError("latest mainline has duplicate industry_code")
    fields = [
        column for column in (
            "industry_name", "mainline_phase", "mainline_phase_gated",
            "strength_percentile", "strength_3d_percentile",
            "turnover_share", "turnover_share_change_1d",
            "turnover_share_change_3d", "startup_evidence_status",
            "peak_evidence", "decay_evidence", "phase_reason",
        ) if column in latest.columns
    ]
    mapped = active.merge(
        latest[["industry_code", *fields]],
        on="industry_code", how="left", validate="many_to_one",
    )
    mapped["mainline_trade_date"] = latest_date
    mapped["research_only"] = True
    mapped["historical_pit_eligible"] = False
    mapped["trading_authorization"] = False
    mapped = mapped.sort_values("symbol").reset_index(drop=True)
    matched = mapped["mainline_phase"].notna() if "mainline_phase" in mapped else pd.Series(False, index=mapped.index)
    audit = {
        "status": "READY_RESEARCH_ONLY" if bool(matched.all()) and len(mapped) else "INCOMPLETE_RESEARCH_COVERAGE",
        "as_of_date": target.date().isoformat(),
        "mainline_trade_date": latest_date.date().isoformat(),
        "active_membership_symbols": int(active["symbol"].nunique()),
        "matched_symbols": int(mapped.loc[matched, "symbol"].nunique()),
        "membership_to_mainline_coverage": float(mapped.loc[matched, "symbol"].nunique() / max(1, active["symbol"].nunique())),
        "mapped_industry_count": int(mapped.loc[matched, "industry_code"].nunique()),
        "historical_pit_eligible": False,
        "trading_authorization": False,
        "research_only": True,
    }
    return mapped, audit
