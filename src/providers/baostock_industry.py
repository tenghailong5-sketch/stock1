"""Baostock current industry membership snapshot.

Baostock can return a broad SH/SZ industry classification snapshot. The
provider's updateDate is retained as source metadata, while available_time is
the actual retrieval time. No historical publication timestamp is inferred.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
import uuid
from datetime import datetime, timezone

import pandas as pd


def _result_frame(result) -> pd.DataFrame:
    if str(getattr(result, "error_code", "")) != "0":
        raise RuntimeError(
            "Baostock query_stock_industry failed: "
            + str(getattr(result, "error_msg", "unknown provider error"))[:240]
        )
    fields = list(getattr(result, "fields", []) or [])
    required = {"updateDate", "code", "code_name", "industry", "industryClassification"}
    if required - set(fields):
        raise ValueError(
            "Baostock industry schema missing: " + ",".join(sorted(required - set(fields)))
        )
    rows = []
    while result.next():
        rows.append(result.get_row_data())
    return pd.DataFrame(rows, columns=fields)


def fetch_baostock_industry_membership_snapshot(
    baostock_module, *, clock=None, login_attempts: int = 3,
    sleep_fn=time.sleep,
) -> pd.DataFrame:
    """Fetch one complete Baostock industry response and stamp retrieval time.

    Transient login/network errors are retried with bounded exponential backoff.
    """
    if login_attempts < 1:
        raise ValueError("login_attempts must be at least 1")
    now = clock or (lambda: datetime.now(timezone.utc))
    login_error = None
    logged_in = False
    for attempt in range(login_attempts):
        try:
            login = baostock_module.login()
            if str(getattr(login, "error_code", "")) == "0":
                logged_in = True
                break
            login_error = RuntimeError(
                "Baostock login failed: "
                + str(getattr(login, "error_msg", "unknown error"))[:240]
            )
        except Exception as exc:
            login_error = exc
        if attempt + 1 < login_attempts:
            try:
                baostock_module.logout()
            except Exception:
                pass
            sleep_fn(2 ** attempt)
    if not logged_in:
        raise RuntimeError(
            "Baostock login failed after "
            f"{login_attempts} attempts: {type(login_error).__name__}: {str(login_error)[:240]}"
        ) from login_error
    try:
        result = baostock_module.query_stock_industry()
        captured = pd.Timestamp(now())
        if captured.tzinfo is None:
            raise ValueError("Baostock observation clock must be timezone-aware")
        captured = captured.tz_convert("UTC")
        frame = _result_frame(result)
    finally:
        baostock_module.logout()

    if frame.empty:
        raise ValueError("Baostock industry snapshot is empty")
    frame = frame.copy()
    frame["code"] = frame["code"].astype(str).str.strip()
    frame["symbol"] = frame["code"].str.extract(r"^(?:sh|sz)\.(\d{6})$", flags=re.IGNORECASE, expand=False)
    if frame["symbol"].isna().any():
        invalid = frame.loc[frame["symbol"].isna(), "code"].head(5).tolist()
        raise ValueError("Baostock industry snapshot contains unsupported security codes: " + ",".join(invalid))
    frame["symbol"] = frame["symbol"].astype(str).str.zfill(6)
    frame["industry"] = frame["industry"].astype(str).str.strip()
    frame["industryClassification"] = frame["industryClassification"].astype(str).str.strip()
    frame["updateDate"] = frame["updateDate"].astype(str).str.strip()
    if frame["updateDate"].eq("").any() or frame["industryClassification"].eq("").any():
        raise ValueError("Baostock industry snapshot contains blank updateDate/classification")
    classifications = set(frame["industryClassification"].unique())
    if len(classifications) != 1:
        raise ValueError("Baostock response mixes industry classification standards")
    # Empty industry values are not assigned to an invented catch-all group.
    frame = frame.loc[frame["industry"].ne("")].copy()
    if frame.empty:
        raise ValueError("Baostock industry snapshot has no classified securities")
    if frame["symbol"].duplicated().any():
        raise ValueError("Baostock industry snapshot contains duplicate securities")

    frame["industry_code"] = (
        frame["industryClassification"] + "::" + frame["industry"]
    )
    frame["industry_name"] = frame["industry"]
    frame["observed_at"] = captured
    frame["available_time"] = captured
    frame["snapshot_id"] = str(uuid.uuid4())
    frame["expected_industry_count"] = int(frame["industry_code"].nunique())
    frame["availability_basis"] = "retrieval_time_only"
    frame["historical_pit_eligible"] = False
    frame["source_id"] = "baostock.query_stock_industry"
    frame["snapshot_scope"] = "current_membership_observation_only"
    revisions = (
        frame.groupby("industry_code", sort=True)[["symbol", "industry_code", "industry_name"]]
        .apply(lambda group: hashlib.sha256(
            json.dumps(
                group.sort_values("symbol").to_dict(orient="records"),
                ensure_ascii=False, sort_keys=True, separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest())
        .to_dict()
    )
    frame["source_revision"] = frame["industry_code"].map(revisions)
    return frame[[
        "symbol", "industry_code", "industry_name", "industryClassification",
        "updateDate", "observed_at", "available_time", "snapshot_id",
        "expected_industry_count", "availability_basis", "historical_pit_eligible",
        "source_id", "snapshot_scope", "source_revision",
    ]].reset_index(drop=True)
