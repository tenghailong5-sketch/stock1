"""Shenwan stock industry reclassification history.

The source provides effective classification dates, but this adapter only
observes the current downloadable workbook at retrieval time. Effective dates
support historical classification research; they do not prove when each
revision was publicly available. Rows therefore remain non-PIT-eligible.
"""
from __future__ import annotations

import hashlib
import uuid
from io import BytesIO
from pathlib import Path
import time

import pandas as pd
import requests


SW_INDUSTRY_HISTORY_URL = (
    "https://www.swsresearch.com/swindex/pdf/SwClass2021/StockClassifyUse_stock.xls"
)
SOURCE_COLUMNS = {
    "code": "股票代码",
    "start_date": "计入日期",
    "industry_code": "行业代码",
    "update_date": "更新日期",
}
OUTPUT_COLUMNS = [
    "snapshot_id", "symbol", "industry_code", "l1_code", "l2_code",
    "valid_from", "valid_to", "source_update_date", "observed_at",
    "available_time", "availability_basis", "source_id", "source_revision",
    "membership_scope", "historical_pit_eligible",
]


def _capture_time(value=None) -> pd.Timestamp:
    stamp = pd.Timestamp.now(tz="UTC") if value is None else pd.Timestamp(value)
    if pd.isna(stamp) or stamp.tzinfo is None:
        raise ValueError("fetched_at must be a valid timezone-aware timestamp")
    return stamp.tz_convert("UTC")


def _normalize_code(values: pd.Series, field: str) -> pd.Series:
    result = values.astype("string").str.strip().str.replace(r"\.0$", "", regex=True)
    result = result.str.zfill(6)
    if result.isna().any() or not result.str.fullmatch(r"\d{6}").all():
        raise ValueError(f"SW industry history contains invalid {field}")
    return result


def normalize_sw_industry_history(
    frame: pd.DataFrame, *, fetched_at=None
) -> pd.DataFrame:
    """Normalize effective-dated SW classifications without claiming PIT lineage."""
    rename = {source: target for target, source in SOURCE_COLUMNS.items()
              if source in frame.columns}
    required = {"股票代码", "计入日期", "行业代码"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError("SW industry history missing columns: " + ",".join(sorted(missing)))

    captured = _capture_time(fetched_at)
    out = frame.rename(columns=rename).copy()
    out["symbol"] = _normalize_code(out["code"], "stock code")
    out["industry_code"] = _normalize_code(out["industry_code"], "industry code")
    out["valid_from"] = pd.to_datetime(out["start_date"], errors="coerce").dt.normalize()
    if out["valid_from"].isna().any():
        raise ValueError("SW industry history contains invalid effective date")

    if "update_date" in out:
        out["source_update_date"] = pd.to_datetime(
            out["update_date"], errors="coerce"
        ).dt.normalize()
    else:
        out["source_update_date"] = pd.NaT

    if out.duplicated(["symbol", "valid_from"]).any():
        raise ValueError("SW industry history contains duplicate symbol/effective date")

    out = out.sort_values(["symbol", "valid_from"]).reset_index(drop=True)
    out["valid_to"] = out.groupby("symbol")["valid_from"].shift(-1)
    out["l1_code"] = out["industry_code"].str[:2] + "0000"
    out["l2_code"] = out["industry_code"].str[:4] + "00"
    out["snapshot_id"] = uuid.uuid4().hex
    out["observed_at"] = captured
    out["available_time"] = captured
    out["availability_basis"] = "retrieval_time_only"
    out["source_id"] = "swsresearch.StockClassifyUse_stock.xls"
    out["source_revision"] = out.apply(
        lambda row: hashlib.sha256(
            "|".join(str(row[field]) for field in
                     ("symbol", "industry_code", "valid_from", "source_update_date"))
            .encode("utf-8")
        ).hexdigest(),
        axis=1,
    )
    out["membership_scope"] = "sw_stock_classification_history"
    out["historical_pit_eligible"] = False
    return out[OUTPUT_COLUMNS].reset_index(drop=True)


def fetch_sw_industry_history(
    *, http_get=None, fetched_at=None, cache_path: str | Path | None = None,
    timeout: int = 35, retry_attempts: int = 2, sleep_fn=time.sleep,
) -> pd.DataFrame:
    """Fetch SW history with pinned-TLS retry and a recent research-only cache fallback."""
    getter = http_get or requests.get
    last_error = None
    payload = None
    for attempt in range(max(1, int(retry_attempts))):
        try:
            try:
                response = getter(
                    SW_INDUSTRY_HISTORY_URL,
                    headers={"User-Agent": "Mozilla/5.0"},
                    timeout=timeout,
                )
                response.raise_for_status()
                payload = response.content
            except requests.exceptions.SSLError:
                # Reuse the already fingerprint-pinned SW certificate adapter.
                from src.providers.sw_industry import _download_xls
                payload = _download_xls(requests, SW_INDUSTRY_HISTORY_URL, timeout)
            frame = pd.read_excel(BytesIO(payload))
            if frame is None or frame.empty:
                raise RuntimeError("SW industry history workbook is empty")
            result = normalize_sw_industry_history(frame, fetched_at=fetched_at)
            result.attrs["provider_status"] = "LIVE"
            if cache_path is not None:
                target = Path(cache_path)
                target.parent.mkdir(parents=True, exist_ok=True)
                result.assign(
                    valid_from=result.valid_from.dt.strftime("%Y-%m-%d"),
                    valid_to=result.valid_to.dt.strftime("%Y-%m-%d"),
                    source_update_date=result.source_update_date.dt.strftime("%Y-%m-%d"),
                    observed_at=result.observed_at.astype(str),
                    available_time=result.available_time.astype(str),
                ).to_json(target, orient="records", lines=True, force_ascii=False)
            return result
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as exc:
            last_error = exc
            if attempt + 1 < max(1, int(retry_attempts)):
                sleep_fn(0.5 * (attempt + 1))
        except Exception as exc:
            last_error = exc
            break

    if cache_path is not None:
        target = Path(cache_path)
        if target.exists():
            try:
                cached = pd.read_json(target, lines=True)
                required = set(OUTPUT_COLUMNS)
                if not cached.empty and required.issubset(cached.columns):
                    cached["valid_from"] = pd.to_datetime(cached.valid_from, errors="coerce")
                    cached["valid_to"] = pd.to_datetime(cached.valid_to, errors="coerce")
                    cached["source_update_date"] = pd.to_datetime(cached.source_update_date, errors="coerce")
                    cached["observed_at"] = pd.to_datetime(cached.observed_at, errors="coerce", utc=True)
                    cached["available_time"] = pd.to_datetime(cached.available_time, errors="coerce", utc=True)
                    cached = cached.dropna(subset=["valid_from", "observed_at", "available_time"])
                    cutoff = pd.Timestamp(fetched_at) if fetched_at is not None else pd.Timestamp.now(tz="UTC")
                    if cutoff.tzinfo is None:
                        raise ValueError("fetched_at must be timezone-aware")
                    latest = cached.observed_at.max()
                    if (cutoff.tz_convert("UTC") - latest).total_seconds() <= 14 * 86400:
                        cached["historical_pit_eligible"] = False
                        cached["availability_basis"] = "retrieval_time_only"
                        cached.attrs["provider_status"] = "CACHED_FALLBACK"
                        cached.attrs["live_fetch_error"] = (
                            f"{type(last_error).__name__}: {str(last_error)[:180]}"
                        )
                        return cached[OUTPUT_COLUMNS].copy()
            except Exception:
                pass
    raise RuntimeError(
        f"SW industry history fetch/parse failed: {type(last_error).__name__}: {str(last_error)[:180]}"
    ) from last_error
