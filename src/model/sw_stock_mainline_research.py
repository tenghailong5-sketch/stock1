"""Research-only link between effective-dated SW stocks and live industry mainlines."""
from __future__ import annotations

import numpy as np
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


def build_forward_stock_mainline_research(
    membership: pd.DataFrame,
    mainline: pd.DataFrame,
    *,
    decision_time,
) -> tuple[pd.DataFrame, dict]:
    """Map the latest retrieval-time SW L1 snapshot to current mainlines.

    This is a live research overlay only. It never claims effective-dated
    history, PIT eligibility, or trading authorization.
    """
    required_membership = {
        "symbol", "industry_level1", "observed_at", "availability_basis",
        "historical_pit_eligible", "source_id", "source_revision", "snapshot_scope",
    }
    required_mainline = {
        "industry_code", "industry_name", "trade_date", "mainline_phase",
        "mainline_phase_gated", "strength_percentile",
    }
    if not isinstance(membership, pd.DataFrame) or not isinstance(mainline, pd.DataFrame):
        raise ValueError("membership and mainline must be DataFrames")
    if membership.empty or mainline.empty:
        return pd.DataFrame(), {
            "status": "NO_CURRENT_MAPPING_INPUTS", "matched_symbols": 0,
            "active_membership_symbols": 0, "historical_pit_eligible": False,
            "trading_authorization": False, "research_only": True,
        }
    missing = required_membership - set(membership.columns)
    if missing:
        raise ValueError("forward membership missing: " + ",".join(sorted(missing)))
    missing = required_mainline - set(mainline.columns)
    if missing:
        raise ValueError("mainline missing: " + ",".join(sorted(missing)))
    cutoff = pd.Timestamp(decision_time)
    if cutoff.tzinfo is None:
        raise ValueError("decision_time must be timezone-aware")
    members = membership.copy()
    members["symbol"] = members["symbol"].astype(str).str.strip().str.zfill(6)
    members["industry_level1"] = members["industry_level1"].astype(str).str.strip()
    members["observed_at"] = pd.to_datetime(members["observed_at"], errors="coerce", utc=True)
    if members["observed_at"].isna().any():
        raise ValueError("forward membership observed_at must be valid and timezone-aware")
    if not members["availability_basis"].astype(str).eq("retrieval_time_only").all():
        raise ValueError("forward membership must remain retrieval_time_only")
    if members["historical_pit_eligible"].map(
        lambda value: value is True or str(value).strip().lower() in {"true", "1", "yes"}
    ).any():
        raise ValueError("PIT-eligible membership is not accepted by forward mapper")
    if not members["snapshot_scope"].astype(str).eq("current_membership_observation_only").all():
        raise ValueError("membership snapshot scope is not current-only")
    members = members.loc[members["observed_at"] <= cutoff.tz_convert("UTC")].copy()
    if members.empty or mainline.empty:
        return pd.DataFrame(), {
            "status": "NO_CURRENT_MAPPING_INPUTS", "matched_symbols": 0,
            "active_membership_symbols": int(members.symbol.nunique()) if not members.empty else 0,
            "historical_pit_eligible": False, "trading_authorization": False,
            "research_only": True,
        }
    latest_snapshot_time = members["observed_at"].max()
    members = members.loc[members["observed_at"].eq(latest_snapshot_time)].copy()
    if "snapshot_id" in members:
        snapshots = members["snapshot_id"].dropna().astype(str).unique()
        if len(snapshots) > 1:
            raise ValueError("latest forward membership time contains multiple snapshots")
    members = members.drop_duplicates(["symbol", "industry_level1"])
    ambiguous = set(members.loc[members.symbol.duplicated(keep=False), "symbol"])
    members = members.loc[~members.symbol.isin(ambiguous)].copy()
    panel = mainline.copy()
    panel["industry_code"] = panel["industry_code"].astype(str).str.strip()
    panel["industry_name"] = panel["industry_name"].astype(str).str.strip()
    panel["trade_date"] = pd.to_datetime(panel["trade_date"], errors="coerce").dt.normalize()
    if panel["trade_date"].isna().any():
        raise ValueError("mainline contains invalid trade_date")
    latest_date = panel["trade_date"].max()
    latest = panel.loc[panel.trade_date.eq(latest_date)].copy()
    if latest.industry_name.duplicated().any():
        raise ValueError("mainline industry names are ambiguous for forward mapping")
    fields = [c for c in (
        "industry_code", "industry_name", "mainline_phase", "mainline_phase_gated",
        "strength_percentile", "strength_3d_percentile", "turnover_share",
        "turnover_share_change_3d", "startup_evidence_status", "peak_evidence",
        "decay_evidence", "phase_gate_reason",
    ) if c in latest.columns]
    mapped = members.merge(
        latest[fields], left_on="industry_level1", right_on="industry_name",
        how="left", validate="many_to_one",
    )
    mapped["membership_observed_at"] = latest_snapshot_time
    mapped["mainline_trade_date"] = latest_date
    mapped["research_only"] = True
    mapped["historical_pit_eligible"] = False
    mapped["trading_authorization"] = False
    mapped = mapped.rename(columns={"industry_level1": "current_industry_level1"})
    mapped = mapped.sort_values("symbol").reset_index(drop=True)
    matched = mapped["mainline_phase"].notna()
    audit = {
        "status": "READY_RESEARCH_ONLY" if bool(matched.all()) and len(mapped) else "INCOMPLETE_RESEARCH_COVERAGE",
        "membership_snapshot_time": latest_snapshot_time.isoformat(),
        "mainline_trade_date": latest_date.date().isoformat(),
        "active_membership_symbols": int(mapped.symbol.nunique()),
        "matched_symbols": int(mapped.loc[matched, "symbol"].nunique()),
        "membership_to_mainline_coverage": float(mapped.loc[matched, "symbol"].nunique() / max(1, mapped.symbol.nunique())),
        "mapped_industry_count": int(mapped.loc[matched, "industry_code"].nunique()),
        "ambiguous_membership_symbols_excluded": len(ambiguous),
        "historical_pit_eligible": False,
        "trading_authorization": False,
        "research_only": True,
    }
    return mapped, audit



def build_spot_industry_strength_panel(
    membership: pd.DataFrame,
    stock_snapshot: pd.DataFrame,
    *,
    trade_date,
    decision_time,
    snapshot_observed_at,
) -> tuple[pd.DataFrame, dict]:
    """Build a one-session sector-strength fallback from a current quote snapshot.

    This fallback can rank current industry strength when the SW index-history
    endpoint is unavailable. It deliberately emits no lifecycle phase: one
    session of constituent quotes cannot prove ignition, peak, or decay.
    """
    required_membership = {
        "symbol", "industry_level1", "observed_at", "availability_basis",
        "historical_pit_eligible", "snapshot_scope",
    }
    required_stock = {"code", "pct_chg", "amount"}
    if not isinstance(membership, pd.DataFrame) or not isinstance(stock_snapshot, pd.DataFrame):
        raise ValueError("membership and stock_snapshot must be DataFrames")
    missing = required_membership - set(membership.columns)
    if missing:
        raise ValueError("spot membership missing: " + ",".join(sorted(missing)))
    missing = required_stock - set(stock_snapshot.columns)
    if missing:
        raise ValueError("spot stock snapshot missing: " + ",".join(sorted(missing)))
    cutoff = pd.Timestamp(decision_time)
    observed = pd.Timestamp(snapshot_observed_at)
    if cutoff.tzinfo is None or observed.tzinfo is None:
        raise ValueError("decision_time and snapshot_observed_at must be timezone-aware")
    cutoff = cutoff.tz_convert("UTC")
    observed = observed.tz_convert("UTC")
    if observed > cutoff:
        raise ValueError("stock quote snapshot is newer than decision_time")
    members = membership.copy()
    members["symbol"] = members["symbol"].astype(str).str.strip().str.zfill(6)
    members["industry_level1"] = members["industry_level1"].astype(str).str.strip()
    members["observed_at"] = pd.to_datetime(members["observed_at"], errors="coerce", utc=True)
    if members["observed_at"].isna().any() or (members["observed_at"] > cutoff).any():
        raise ValueError("membership observation time is invalid or after decision_time")
    if not members["availability_basis"].astype(str).eq("retrieval_time_only").all():
        raise ValueError("spot membership must remain retrieval_time_only")
    if members["historical_pit_eligible"].map(
        lambda value: value is True or str(value).strip().lower() in {"true", "1", "yes"}
    ).any():
        raise ValueError("PIT-eligible membership is not accepted by spot fallback")
    if not members["snapshot_scope"].astype(str).eq("current_membership_observation_only").all():
        raise ValueError("spot membership scope must be current-only")
    latest_snapshot = members["observed_at"].max()
    members = members.loc[members["observed_at"].eq(latest_snapshot)].copy()
    members = members.drop_duplicates(["symbol", "industry_level1"])
    ambiguous = set(members.loc[members.symbol.duplicated(keep=False), "symbol"])
    members = members.loc[~members.symbol.isin(ambiguous)].copy()

    stocks = stock_snapshot.copy()
    stocks["symbol"] = stocks["code"].astype(str).str.strip().str[-6:].str.zfill(6)
    stocks["pct_chg"] = pd.to_numeric(stocks["pct_chg"], errors="coerce")
    stocks["amount"] = pd.to_numeric(stocks["amount"], errors="coerce")
    stocks = stocks.loc[
        stocks["symbol"].str.fullmatch(r"\d{6}", na=False)
        & np.isfinite(stocks["pct_chg"])
        & np.isfinite(stocks["amount"])
        & stocks["amount"].gt(0)
    ].copy()
    if stocks["symbol"].duplicated().any():
        raise ValueError("spot stock snapshot contains duplicate symbols")
    joined = members.merge(
        stocks[["symbol", "pct_chg", "amount"]],
        on="symbol", how="inner", validate="one_to_one",
    )
    if joined.empty:
        return pd.DataFrame(), {
            "status": "NO_MATCHED_SPOT_STOCKS", "matched_symbols": 0,
            "ambiguous_membership_symbols_excluded": len(ambiguous),
            "mainline_phase_available": False,
            "historical_pit_eligible": False, "trading_authorization": False,
            "research_only": True,
        }
    joined["amount_weighted_return"] = joined["pct_chg"] * joined["amount"]
    panel = joined.groupby("industry_level1", as_index=False).agg(
        industry_amount=("amount", "sum"),
        amount_weighted_return=("amount_weighted_return", "sum"),
        mean_stock_return_pct=("pct_chg", "mean"),
        median_stock_return_pct=("pct_chg", "median"),
        positive_breadth=("pct_chg", lambda values: float((values > 0).mean())),
        stock_count=("symbol", "nunique"),
    )
    panel["amount_weighted_return_pct"] = (
        panel["amount_weighted_return"] / panel["industry_amount"]
    )
    panel["strength_percentile"] = panel["amount_weighted_return_pct"].rank(
        method="average", pct=True
    )
    total_amount = float(panel["industry_amount"].sum())
    panel["turnover_share"] = panel["industry_amount"] / total_amount
    panel["industry_code"] = "SWL1:" + panel["industry_level1"].astype(str)
    panel["industry_name"] = panel["industry_level1"].astype(str)
    panel["trade_date"] = pd.Timestamp(trade_date).normalize()
    panel["mainline_phase"] = "UNKNOWN"
    panel["mainline_phase_gated"] = "UNKNOWN"
    panel["startup_evidence_status"] = "UNKNOWN"
    panel["phase_gate_reason"] = "one_session_spot_only_no_lifecycle_history"
    panel["strength_basis"] = "amount_weighted_current_constituent_return"
    panel["mainline_phase_available"] = False
    panel["research_only"] = True
    panel["historical_pit_eligible"] = False
    panel["trading_authorization"] = False
    panel = panel.sort_values(
        ["strength_percentile", "industry_name"], ascending=[False, True]
    ).reset_index(drop=True)
    audit = {
        "status": "READY_SPOT_ONLY_RESEARCH",
        "trade_date": pd.Timestamp(trade_date).date().isoformat(),
        "membership_snapshot_time": latest_snapshot.isoformat(),
        "stock_snapshot_observed_at": observed.isoformat(),
        "matched_symbols": int(joined.symbol.nunique()),
        "industry_count": int(panel.industry_code.nunique()),
        "ambiguous_membership_symbols_excluded": len(ambiguous),
        "mainline_phase_available": False,
        "historical_pit_eligible": False,
        "trading_authorization": False,
        "research_only": True,
    }
    return panel, audit
