"""Official Shenwan 2021 current A-share industry membership snapshot.

Downloads the two public XLS files published by Shenwan Hongyuan Research:
- StockClassifyUse_stock.xls: stock-to-industry change history
- SwClassCode_2021.xls: industry-code/name hierarchy

The latest entry date per stock is selected. Availability is stamped at retrieval
time only; this is not a historical point-in-time publication feed.
"""
from __future__ import annotations

import hashlib
import ssl
import tempfile
import uuid
from datetime import datetime, timezone

import pandas as pd
import requests
import xlrd


BASE_URL = "https://www.swsresearch.com/swindex/pdf/SwClass2021"
STOCK_URL = f"{BASE_URL}/StockClassifyUse_stock.xls"
CODE_URL = f"{BASE_URL}/SwClassCode_2021.xls"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; FHM100/1.0)"}
INTERMEDIATE_URL = (
    "https://cacerts.digicert.com/"
    "DigiCertBasicOVG2TLSCNRSA4096SHA2562022CA1.crt"
)
INTERMEDIATE_SHA1 = "60707270F2100EE2B771FEC9EFFAD8C9BFFE3358"


def _trusted_intermediate_bundle(*, timeout: int = 15) -> str:
    """Append the pinned SW-host intermediate CA to certifi's normal trust roots."""
    response = requests.get(INTERMEDIATE_URL, timeout=timeout)
    response.raise_for_status()
    payload = response.content
    if payload.startswith(b"-----BEGIN CERTIFICATE-----"):
        pem = payload
        der = ssl.PEM_cert_to_DER_cert(payload.decode("ascii"))
    else:
        der = payload
        pem = ssl.DER_cert_to_PEM_cert(payload).encode("ascii")
    fingerprint = hashlib.sha1(der).hexdigest().upper()
    if fingerprint != INTERMEDIATE_SHA1:
        raise RuntimeError(
            "DigiCert intermediate certificate fingerprint mismatch: " + fingerprint
        )
    base_bundle = requests.certs.where()
    with open(base_bundle, "rb") as handle:
        base_pem = handle.read()
    handle = tempfile.NamedTemporaryFile(
        prefix="fhm-sw-xls-ca-", suffix=".pem", delete=False
    )
    try:
        handle.write(base_pem)
        handle.write(b"\\n")
        handle.write(pem)
        handle.flush()
        return handle.name
    finally:
        handle.close()


def _download_xls(client, url: str, timeout: int) -> bytes:
    """Download an XLS with verified TLS, adding a pinned intermediate CA if needed.

    Never disable certificate verification or trust an unpinned certificate.
    """
    try:
        response = client.get(url, headers=HEADERS, timeout=timeout)
    except requests.exceptions.SSLError:
        # The official SW host currently omits its intermediate CA in the
        # default chain. Reuse a pinned DigiCert intermediate; never disable TLS.
        bundle = _trusted_intermediate_bundle(timeout=min(timeout, 15))
        response = client.get(
            url, headers=HEADERS, timeout=timeout, verify=bundle
        )
    response.raise_for_status()
    content = response.content
    # Legacy Excel .xls files are OLE compound documents. Reject HTML error pages
    # and proxy login pages before handing their bytes to xlrd.
    if not content or not content.startswith(bytes.fromhex("D0CF11E0A1B11AE1")):
        raise ValueError(f"Shenwan XLS download is not a valid OLE .xls file: {url}")
    return content


def _text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def _sheet_rows(content: bytes):
    workbook = xlrd.open_workbook(file_contents=content)
    if workbook.nsheets < 1:
        raise ValueError("Shenwan XLS has no worksheets")
    sheet = workbook.sheet_by_index(0)
    if sheet.nrows < 2:
        raise ValueError("Shenwan XLS worksheet is empty")
    headers = [_text(value) for value in sheet.row_values(0)]
    return headers, (sheet.row_values(i) for i in range(1, sheet.nrows))


def _column(headers: list[str], name: str) -> int:
    try:
        return headers.index(name)
    except ValueError as exc:
        raise ValueError(
            f"Shenwan XLS missing required column {name!r}; got {headers}"
        ) from exc


def fetch_sw_current_industry_membership_snapshot(
    *, clock=None, session=None, timeout: int = 60
) -> pd.DataFrame:
    """Download official SW2021 tables and select each stock's latest industry."""
    now = clock or (lambda: datetime.now(timezone.utc))
    client = session or requests
    stock_content = _download_xls(client, STOCK_URL, timeout)
    code_content = _download_xls(client, CODE_URL, timeout)

    code_headers, code_rows = _sheet_rows(code_content)
    code_idx = _column(code_headers, "行业代码")
    l1_idx = _column(code_headers, "一级行业名称")
    l2_idx = _column(code_headers, "二级行业名称")
    l3_idx = _column(code_headers, "三级行业名称")
    code_map = {}
    for row in code_rows:
        code = _text(row[code_idx]).zfill(6)
        names = (_text(row[l1_idx]), _text(row[l2_idx]), _text(row[l3_idx]))
        if code and all(names):
            code_map[code] = names
    if not code_map:
        raise ValueError("Shenwan industry-code dictionary contains no usable rows")

    stock_headers, stock_rows = _sheet_rows(stock_content)
    symbol_idx = _column(stock_headers, "股票代码")
    date_idx = _column(stock_headers, "计入日期")
    industry_idx = _column(stock_headers, "行业代码")

    latest: dict[str, tuple[float, str]] = {}
    for row in stock_rows:
        symbol = _text(row[symbol_idx])
        industry_code = _text(row[industry_idx]).zfill(6)
        try:
            entry_date = float(row[date_idx])
        except (TypeError, ValueError):
            continue
        if len(symbol) != 6 or not symbol.isdigit() or industry_code not in code_map:
            continue
        current = latest.get(symbol)
        if current is None or entry_date > current[0]:
            latest[symbol] = (entry_date, industry_code)

    if not latest:
        raise ValueError("Shenwan stock classification file contains no current SW2021 members")

    captured = pd.Timestamp(now())
    if captured.tzinfo is None:
        raise ValueError("Shenwan observation clock must be timezone-aware")
    captured = captured.tz_convert("UTC")
    snapshot_id = str(uuid.uuid4())
    rows = []
    for symbol, (entry_date, industry_code) in sorted(latest.items()):
        level1, level2, level3 = code_map[industry_code]
        revision = hashlib.sha256(
            f"{symbol}|{industry_code}|{level1}|{level2}|{level3}|{entry_date}".encode("utf-8")
        ).hexdigest()
        rows.append({
            "symbol": symbol,
            "industry_code": industry_code,
            "industry_name": level3,
            "industry_level1": level1,
            "industry_level2": level2,
            "snapshot_id": snapshot_id,
            "observed_at": captured,
            "available_time": captured,
            "availability_basis": "retrieval_time_only",
            "historical_pit_eligible": False,
            "source_id": "swsresearch.sw2021_stock_classification",
            "snapshot_scope": "current_membership_observation_only",
            "source_revision": revision,
        })
    result = pd.DataFrame(rows)
    if result["symbol"].duplicated().any():
        raise ValueError("Shenwan current snapshot contains duplicate stock symbols")
    result["expected_industry_count"] = int(result["industry_code"].nunique())
    return result[[
        "symbol", "industry_code", "industry_name", "industry_level1",
        "industry_level2", "snapshot_id", "expected_industry_count",
        "observed_at", "available_time", "availability_basis",
        "historical_pit_eligible", "source_id", "snapshot_scope",
        "source_revision",
    ]]
