"""Forward-only current SW industry membership via the official SW index API.

TLS verification stays enabled. If the SW host omits its intermediate CA, the
adapter downloads the published DigiCert intermediate, checks its pinned SHA-1
fingerprint, and adds it to a temporary copy of the normal trust bundle.
"""
from __future__ import annotations

import hashlib
import json
import math
import ssl
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests


INDEX_URL = "https://www.swsresearch.com/institute-sw/api/index_publish/current/"
COMPONENT_URL = "https://www.swsresearch.com/institute-sw/api/index_publish/details/component_stocks/"
INTERMEDIATE_URL = (
    "https://cacerts.digicert.com/"
    "DigiCertBasicOVG2TLSCNRSA4096SHA2562022CA1.crt"
)
# Published SHA-1 certificate thumbprint from DigiCert/Microsoft CA-chain records.
INTERMEDIATE_SHA1 = "60707270F2100EE2B771FEC9EFFAD8C9BFFE3358"
GEOTRUST_INTERMEDIATE_URL = (
    "http://cacerts.digicert.cn/"
    "GeoTrustG2TLSCNRSA4096SHA2562022CA1.crt"
)
GEOTRUST_INTERMEDIATE_SHA256 = "05DC9EDC0FDDF A975A1432EF806EC780078B5362AD45AF76DB15C907630DB25D".replace(" ", "")
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Referer": "https://www.swsresearch.com/institute_sw/allIndex/releasedIndex",
    "Accept": "application/json, text/plain, */*",
}


def _trusted_intermediate_bundle(*, timeout: int = 15) -> str:
    """Build a certifi bundle with pinned SW-host intermediates; never disable TLS."""
    base_bundle = Path(requests.certs.where()).read_bytes()
    pinned = [
        (INTERMEDIATE_URL, "sha1", INTERMEDIATE_SHA1),
        (GEOTRUST_INTERMEDIATE_URL, "sha256", GEOTRUST_INTERMEDIATE_SHA256),
    ]
    certificates = []
    failures = []
    for url, algorithm, expected in pinned:
        try:
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()
            payload = response.content
            if payload.startswith(b"-----BEGIN CERTIFICATE-----"):
                pem = payload
                der = ssl.PEM_cert_to_DER_cert(payload.decode("ascii"))
            else:
                der = payload
                pem = ssl.DER_cert_to_PEM_cert(payload).encode("ascii")
            actual = hashlib.sha1(der).hexdigest().upper() if algorithm == "sha1" else hashlib.sha256(der).hexdigest().upper()
            if actual != expected:
                raise RuntimeError("pinned SW intermediate certificate fingerprint mismatch: " + actual)
            certificates.append(pem)
        except Exception as exc:
            failures.append(f"{url}: {type(exc).__name__}: {exc}")
    if not certificates:
        raise RuntimeError("unable to retrieve any pinned SW intermediate: " + " | ".join(failures))
    handle = tempfile.NamedTemporaryFile(
        prefix="fhm-sw-ca-", suffix=".pem", delete=False
    )
    try:
        handle.write(base_bundle)
        for pem in certificates:
            handle.write(b"\n")
            handle.write(pem)
        handle.flush()
        return handle.name
    finally:
        handle.close()


def _get_json(session, url: str, *, params: dict, timeout: int):
    try:
        response = session.get(
            url, params=params, headers=HEADERS, timeout=timeout
        )
    except requests.exceptions.SSLError:
        bundle = getattr(session, "_fhm_sw_ca_bundle", None)
        if bundle is None:
            bundle = _trusted_intermediate_bundle(timeout=timeout)
            session._fhm_sw_ca_bundle = bundle
            session.verify = bundle
        response = session.get(
            url, params=params, headers=HEADERS, timeout=timeout
        )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict) or not isinstance(payload.get("data"), dict):
        raise ValueError("SW API response missing data object")
    return payload["data"]


def _paged_results(session, url: str, *, params: dict, timeout: int) -> list[dict]:
    first = _get_json(session, url, params={**params, "page": 1}, timeout=timeout)
    results = first.get("results")
    if not isinstance(results, list):
        raise ValueError("SW API response missing results list")
    count = int(first.get("count") or len(results))
    page_size = int(params["page_size"])
    total_pages = max(1, math.ceil(count / page_size))
    rows = list(results)
    for page in range(2, total_pages + 1):
        data = _get_json(
            session, url, params={**params, "page": page}, timeout=timeout
        )
        batch = data.get("results")
        if not isinstance(batch, list):
            raise ValueError(f"SW API page {page} missing results list")
        rows.extend(batch)
    return rows


def fetch_sw_api_industry_membership_snapshot(
    *, session=None, clock=None, timeout: int = 25, pause_seconds: float = 0.3,
    sleep_fn=time.sleep, industry_level: str = "一级行业",
) -> pd.DataFrame:
    """Fetch current SW industry constituents, preserving retrieval-time lineage.

    The API returns current constituents, not a historical point-in-time ledger.
    """
    if industry_level not in {"一级行业", "二级行业"}:
        raise ValueError("industry_level must be 一级行业 or 二级行业")
    client = session or requests.Session()
    now = clock or (lambda: datetime.now(timezone.utc))

    groups = _paged_results(
        client, INDEX_URL,
        params={"page_size": 50, "indextype": industry_level},
        timeout=timeout,
    )
    groups = [
        row for row in groups
        if isinstance(row, dict) and str(row.get("swindexcode") or "").strip()
        and str(row.get("swindexname") or "").strip()
    ]
    if not groups:
        raise RuntimeError("SW API returned no industry index groups")

    snapshot_id = str(uuid.uuid4())
    rows = []
    failures = []
    for index, group in enumerate(groups):
        code = str(group["swindexcode"]).strip()
        name = str(group["swindexname"]).strip()
        try:
            members = _paged_results(
                client, COMPONENT_URL,
                params={
                    "swindexcode": code,
                    "page_size": 10000,
                },
                timeout=timeout,
            )
            captured = pd.Timestamp(now())
            if captured.tzinfo is None:
                raise ValueError("SW observation clock must be timezone-aware")
            captured = captured.tz_convert("UTC")
            clean = []
            for item in members:
                raw_symbol = str(item.get("stockcode") or "").strip()
                if not raw_symbol.isdigit() or len(raw_symbol) > 6:
                    continue
                symbol = raw_symbol.zfill(6)
                clean.append({
                    "symbol": symbol,
                    "industry_code": code,
                    "industry_name": name,
                })
            if not clean:
                failures.append(code)
                continue
            revision = hashlib.sha256(
                json.dumps(clean, ensure_ascii=False, sort_keys=True).encode("utf-8")
            ).hexdigest()
            for item in clean:
                rows.append({
                    **item,
                    "industry_level1": name if industry_level == "一级行业" else None,
                    "industry_level2": name if industry_level == "二级行业" else None,
                    "snapshot_id": snapshot_id,
                    "expected_industry_count": len(groups),
                    "observed_at": captured,
                    "available_time": captured,
                    "availability_basis": "retrieval_time_only",
                    "historical_pit_eligible": False,
                    "source_id": "swsresearch.index_publish.component_stocks",
                    "source_taxonomy": "SW",
                    "source_taxonomy_version": "publisher_current_unversioned",
                    "classification_level": industry_level,
                    "snapshot_scope": "current_membership_observation_only",
                    "source_revision": revision,
                })
        except Exception:
            failures.append(code)
        if pause_seconds > 0 and index + 1 < len(groups):
            sleep_fn(pause_seconds)

    if not rows:
        raise RuntimeError(
            "SW API returned no usable industry constituents; "
            f"failed_groups={len(failures)}/{len(groups)}"
        )
    frame = pd.DataFrame(rows)
    if frame.duplicated(["symbol", "industry_code"]).any():
        frame = frame.drop_duplicates(["symbol", "industry_code"], keep="first")
    frame.attrs["requested_industry_count"] = len(groups)
    frame.attrs["failed_industry_codes"] = sorted(set(failures))
    frame.attrs["industry_level"] = industry_level
    return frame


TREND_URL = "https://www.swsresearch.com/institute-sw/api/index_publish/trend/"


def _get_json_payload(session, url: str, *, params: dict, timeout: int):
    """GET a SW API JSON payload while preserving TLS verification."""
    try:
        response = session.get(url, params=params, headers=HEADERS, timeout=timeout)
    except requests.exceptions.SSLError:
        bundle = getattr(session, "_fhm_sw_ca_bundle", None)
        if bundle is None:
            bundle = _trusted_intermediate_bundle(timeout=timeout)
            session._fhm_sw_ca_bundle = bundle
            session.verify = bundle
        response = session.get(url, params=params, headers=HEADERS, timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict) or "data" not in payload:
        raise ValueError("SW API response missing data payload")
    return payload["data"]


def fetch_sw_api_first_level_index_list(*, session=None, timeout: int = 25) -> pd.DataFrame:
    """Fetch the current SW first-level index-code list from the official API."""
    client = session or requests.Session()
    rows = _paged_results(
        client, INDEX_URL,
        params={"page_size": 50, "indextype": "一级行业"},
        timeout=timeout,
    )
    clean = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        code = str(row.get("swindexcode") or "").strip()
        name = str(row.get("swindexname") or "").strip()
        if len(code) == 6 and code.isdigit() and name:
            clean.append({"industry_code": code, "industry_name": name})
    frame = pd.DataFrame(clean).drop_duplicates("industry_code")
    if len(frame) < 20:
        raise RuntimeError(f"SW first-level index list unexpectedly small: {len(frame)}")
    return frame.sort_values("industry_code").reset_index(drop=True)


def fetch_sw_api_industry_index_history(
    symbol: str, *, start_date: str, end_date: str,
    session=None, clock=None, timeout: int = 25,
) -> pd.DataFrame:
    """Fetch official SW daily industry-index history using verified TLS.

    The API supplies historical index dates but not original historical
    publication/availability times. Every row is therefore retrieval-time-only
    and explicitly excluded from historical PIT scoring.
    """
    code = str(symbol).strip()
    if len(code) != 6 or not code.isdigit():
        raise ValueError("SW industry index symbol must be a six-digit code")
    for value, label in ((start_date, "start_date"), (end_date, "end_date")):
        parsed = pd.to_datetime(value, format="%Y%m%d", errors="coerce")
        if pd.isna(parsed) or parsed.strftime("%Y%m%d") != value:
            raise ValueError(label + " must use YYYYMMDD")
    if start_date > end_date:
        raise ValueError("start_date cannot be after end_date")
    client = session or requests.Session()
    payload = _get_json_payload(
        client, TREND_URL,
        params={"swindexcode": code, "period": "DAY"},
        timeout=timeout,
    )
    if not isinstance(payload, list) or not payload:
        raise ValueError("SW industry index history payload is empty or invalid")
    frame = pd.DataFrame(payload).rename(columns={
        "bargaindate": "trade_date",
        "openindex": "open",
        "maxindex": "high",
        "minindex": "low",
        "closeindex": "close",
        "bargainamount": "volume",
        "bargainsum": "amount",
    })
    required = {"trade_date", "close", "amount"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError("SW index history missing fields: " + ",".join(sorted(missing)))
    frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce").dt.normalize()
    for column in ("open", "high", "low", "close", "volume", "amount"):
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["trade_date", "close", "amount"]).sort_values("trade_date")
    frame = frame.loc[
        frame.trade_date.between(pd.Timestamp(start_date), pd.Timestamp(end_date))
    ].copy()
    if frame.empty or frame["trade_date"].duplicated().any():
        raise ValueError("SW industry index history has no valid rows or duplicate dates")
    if (frame["close"] <= 0).any() or (frame["amount"] < 0).any():
        raise ValueError("SW industry index history contains invalid close or amount")
    captured = pd.Timestamp(clock() if clock else datetime.now(timezone.utc))
    if captured.tzinfo is None:
        raise ValueError("SW industry index retrieval clock must be timezone-aware")
    captured = captured.tz_convert("UTC")
    frame["industry_code"] = code
    frame["daily_return"] = frame["close"].pct_change()
    frame["return_5"] = frame["close"].div(frame["close"].shift(5)).sub(1)
    frame["return_20"] = frame["close"].div(frame["close"].shift(20)).sub(1)
    frame["available_time"] = captured
    frame["observed_at"] = captured
    frame["availability_basis"] = "retrieval_time_only"
    frame["historical_pit_eligible"] = False
    frame["source_id"] = "swsresearch.index_publish.trend"
    frame["source_revision"] = frame.apply(
        lambda row: hashlib.sha256(
            f"{code}|{row['trade_date']}|{row['close']}|{row['amount']}".encode("utf-8")
        ).hexdigest(), axis=1,
    )
    frame["snapshot_scope"] = "historical_index_series_retrieved_at_current_time"
    return frame.reset_index(drop=True)
