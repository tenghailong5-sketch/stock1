"""CNINFO current industry membership adapter (forward-only snapshot).

CNINFO p_stock2110 exposes classification-change rows per security.  This
adapter keeps the CSRC classification only and selects the latest available
row per symbol.  It is deliberately slow/rate-limited and never backdates
current membership into historical point-in-time data.
"""
from __future__ import annotations

import time
from datetime import datetime, timezone

import pandas as pd


SOURCE_ID = "cninfo.p_stock2110.csrc"


def fetch_cninfo_current_industry_membership_snapshot(
    ak_module, symbols, *, clock=None, pause_seconds: float = 0.25,
    sleep_fn=time.sleep, start_date: str = "19900101", end_date: str | None = None,
) -> pd.DataFrame:
    """Fetch CSRC classification for requested symbols, one CNINFO query each."""
    now = clock or (lambda: datetime.now(timezone.utc))
    end_date = end_date or datetime.now(timezone.utc).strftime("%Y%m%d")
    requested = sorted({
        str(value).strip().extract if False else str(value).strip()
        for value in symbols
    })
    requested = sorted({
        code.zfill(6) for value in requested
        if (code := "".join(ch for ch in value if ch.isdigit())) and len(code) <= 6
    })
    rows, failures = [], {}
    provider = getattr(ak_module, "stock_industry_change_cninfo", None)
    if not callable(provider):
        raise RuntimeError("AKShare stock_industry_change_cninfo is unavailable")

    for symbol in requested:
        try:
            frame = provider(
                symbol=symbol, start_date=start_date, end_date=end_date
            )
            captured = pd.Timestamp(now())
            if captured.tzinfo is None:
                raise ValueError("CNINFO capture clock must be timezone-aware")
            if frame is None or frame.empty:
                failures[symbol] = "empty_response"
            else:
                required = {"分类标准编码", "分类标准", "行业编码", "变更日期"}
                missing = required - set(frame.columns)
                if missing:
                    raise ValueError("CNINFO schema missing: " + ",".join(sorted(missing)))
                work = frame.copy()
                standard = work["分类标准编码"].astype(str).str.strip()
                standard_name = work["分类标准"].astype(str)
                selected = work.loc[
                    standard.eq("008001") | standard_name.str.contains("证监会", na=False)
                ].copy()
                if selected.empty:
                    failures[symbol] = "no_csrc_classification"
                else:
                    selected["变更日期"] = pd.to_datetime(
                        selected["变更日期"], errors="coerce"
                    )
                    selected = selected.sort_values("变更日期").tail(1)
                    rec = selected.iloc[0]
                    industry_code = str(rec["行业编码"]).strip()
                    name_candidates = [
                        rec.get("行业中类"), rec.get("行业大类"),
                        rec.get("行业次类"), rec.get("行业门类"),
                    ]
                    industry_name = next(
                        (str(value).strip() for value in name_candidates
                         if pd.notna(value) and str(value).strip()),
                        "",
                    )
                    if not industry_code or industry_code.lower() == "nan" or not industry_name:
                        failures[symbol] = "invalid_latest_classification"
                    else:
                        rows.append({
                            "symbol": symbol,
                            "industry_code": industry_code,
                            "industry_name": industry_name,
                            "observed_at": captured,
                            "available_time": captured,
                            "source_id": SOURCE_ID,
                            "snapshot_scope": "current_membership_observation_only",
                            "availability_basis": "retrieval_time_only",
                            "historical_pit_eligible": False,
                            "taxonomy": "CSRC",
                        })
        except Exception as exc:
            failures[symbol] = f"{type(exc).__name__}: {str(exc)[:160]}"
        if pause_seconds > 0:
            sleep_fn(pause_seconds)

    result = pd.DataFrame(rows)
    if not result.empty:
        result["snapshot_id"] = f"cninfo-{pd.Timestamp(now()).strftime('%Y%m%dT%H%M%S')}"
        result["expected_industry_count"] = int(result["industry_code"].nunique())
        result["source_revision"] = (
            result[["symbol", "industry_code", "industry_name"]]
            .astype(str).agg("|".join, axis=1)
            .map(lambda value: __import__("hashlib").sha256(value.encode()).hexdigest())
        )
        if result["symbol"].duplicated().any():
            raise ValueError("CNINFO returned duplicate current classifications")
    result.attrs["failed_symbols"] = failures
    result.attrs["requested_symbol_count"] = len(requested)
    return result
