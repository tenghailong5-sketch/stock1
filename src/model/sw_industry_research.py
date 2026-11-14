"""Research-only effective-date overlay for current Shenwan stock classifications."""
from __future__ import annotations

import pandas as pd


def select_sw_industry_research_snapshot(
    history: pd.DataFrame, expected_symbols, *, as_of_date
) -> tuple[pd.DataFrame, dict]:
    """Select effective-date matches for research display, never PIT authorization."""
    required = {
        "symbol", "industry_code", "l1_code", "l2_code", "valid_from",
        "valid_to", "available_time", "availability_basis",
        "historical_pit_eligible",
    }
    if history is None or not isinstance(history, pd.DataFrame):
        raise ValueError("history must be a DataFrame")
    missing = required - set(history.columns)
    if missing:
        raise ValueError("SW history missing columns: " + ",".join(sorted(missing)))
    symbols = [str(value).strip().zfill(6) for value in expected_symbols]
    if not symbols or any(not value for value in symbols):
        raise ValueError("expected_symbols must be non-empty")
    if len(symbols) != len(set(symbols)):
        raise ValueError("expected_symbols contains duplicates")
    target = pd.Timestamp(as_of_date).normalize()
    if pd.isna(target):
        raise ValueError("as_of_date must be valid")

    x = history.copy()
    x["symbol"] = x["symbol"].astype(str).str.strip().str.zfill(6)
    x["valid_from"] = pd.to_datetime(x["valid_from"], errors="coerce").dt.normalize()
    x["valid_to"] = pd.to_datetime(x["valid_to"], errors="coerce").dt.normalize()
    available = pd.to_datetime(x["available_time"], errors="coerce", utc=True)
    if x["valid_from"].isna().any() or available.isna().any():
        raise ValueError("SW history dates and availability must be valid")
    if x["historical_pit_eligible"].map(
        lambda value: value is True or str(value).strip().lower() in {"true", "1", "yes"}
    ).any():
        raise ValueError("PIT-eligible rows are not accepted by research-only overlay")
    if not x["availability_basis"].astype(str).eq("retrieval_time_only").all():
        raise ValueError("SW history must retain retrieval_time_only provenance")

    selected = x.loc[
        (x["valid_from"] <= target)
        & (x["valid_to"].isna() | (target < x["valid_to"]))
        & x["symbol"].isin(symbols)
    ].copy()
    if selected["symbol"].duplicated().any():
        selected = selected.iloc[0:0].copy()
        status = "ambiguous_effective_membership"
    else:
        status = "ready_research_only" if set(selected.symbol) == set(symbols) else "incomplete_research_coverage"
    selected["research_only"] = True
    selected["trading_authorization"] = False
    selected = selected.sort_values("symbol").reset_index(drop=True)
    audit = {
        "status": status,
        "as_of_date": target.date().isoformat(),
        "expected_symbols": len(symbols),
        "matched_symbols": int(selected.symbol.nunique()),
        "coverage": float(selected.symbol.nunique() / len(symbols)),
        "historical_pit_eligible": False,
        "trading_authorization": False,
        "availability_basis": "retrieval_time_only",
    }
    return selected, audit
