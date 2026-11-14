"""Official TongDaXin daily package adapter for full-market A-share bars.

The package is a daily market snapshot (SSE/SZSE/BSE). Retrieval time is
recorded separately from trade_date; downloaded historical packages are NOT
historical point-in-time evidence because original publication/revision times
are not supplied by the publisher.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import re
import struct
import zipfile
import zlib
from datetime import datetime, timezone

import pandas as pd
import requests

TDX_PACKAGE_URL = "https://www.tdx.com.cn/products/data/data/g4day/{ymd}.zip"
TDX_BJ_FIRST_DAY = "20220506"
TDX_MIN_PRICED = {"sh": 10000, "sz": 3000, "bj": 50}
REQUIRED_COLUMNS = {
    "trade_date", "market", "symbol", "name", "prev_close", "open",
    "high", "low", "close", "volume", "amount", "available_time",
    "source_id", "source_revision", "availability_basis",
    "historical_pit_eligible",
}


def _trade_day(value) -> str:
    day = pd.Timestamp(value)
    if pd.isna(day):
        raise ValueError("trade_date is invalid")
    return day.strftime("%Y%m%d")


def _capture_time(value=None) -> pd.Timestamp:
    stamp = pd.Timestamp.now(tz="UTC") if value is None else pd.Timestamp(value)
    if stamp.tzinfo is None:
        raise ValueError("available_time must be timezone-aware")
    return stamp.tz_convert("UTC")


def _parse_package(content: bytes, ymd: str, *, min_priced=None) -> list[dict]:
    if not content.startswith(b"PK"):
        raise RuntimeError("TongDaXin response is not a ZIP package")
    try:
        archive = zipfile.ZipFile(io.BytesIO(content))
    except (zipfile.BadZipFile, zlib.error, EOFError) as exc:
        raise RuntimeError(f"TongDaXin ZIP is corrupt: {exc}") from exc

    names = set(archive.namelist())
    thresholds = dict(TDX_MIN_PRICED)
    if min_priced is not None:
        thresholds.update(min_priced)
    rows: list[dict] = []
    for market in ("sh", "sz", "bj"):
        cod_name, md1_name = f"{market}{ymd[2:]}.cod", f"{market}{ymd[2:]}.md1"
        if market == "bj" and ymd < TDX_BJ_FIRST_DAY and cod_name not in names and md1_name not in names:
            continue
        if cod_name not in names or md1_name not in names:
            raise RuntimeError(f"package missing {cod_name}/{md1_name}")
        cod, md1 = archive.read(cod_name), archive.read(md1_name)
        if len(cod) % 150 or len(md1) % 512:
            raise RuntimeError(f"{market} code/quote block has invalid byte length")
        if len(cod) // 150 != len(md1) // 512:
            raise RuntimeError(f"{market} code and quote block counts differ")
        before = len(rows)
        codes, seqs = set(), set()
        for offset in range(0, len(cod), 150):
            record = cod[offset:offset + 150]
            try:
                code = record[:6].rstrip(b"\\x00 ").decode("ascii")
            except UnicodeDecodeError as exc:
                raise RuntimeError(f"{market} contains a non-ASCII security code") from exc
            seq = struct.unpack("<H", record[32:34])[0]
            if not re.fullmatch(r"[0-9]{6}", code) or code in codes or seq in seqs:
                raise RuntimeError(f"{market} contains invalid/duplicate code or block index")
            codes.add(code)
            seqs.add(seq)
            block = md1[seq * 512:(seq + 1) * 512]
            if len(block) != 512:
                raise RuntimeError(f"{market}{code} quote block is out of range")
            prev_close = struct.unpack("<d", block[4:12])[0]
            open_, high, low, close = struct.unpack("<4d", block[12:44])
            volume = struct.unpack("<Q", block[56:64])[0]
            amount = struct.unpack("<d", block[72:80])[0]
            values = (prev_close, open_, high, low, close, amount)
            if not all(math.isfinite(v) for v in values):
                raise RuntimeError(f"{market}{code} contains non-finite market values")
            if close <= 0:
                continue
            if min(prev_close, open_, high, low) < 0 or amount < 0 or volume < 0:
                raise RuntimeError(f"{market}{code} contains negative price/volume/amount")
            if high < max(open_, low, close) or low > min(open_, high, close):
                raise RuntimeError(f"{market}{code} has inconsistent OHLC")
            raw_name = record[40:72].split(b"\\x00")[0]
            try:
                name = raw_name.decode("gbk").strip()
            except UnicodeDecodeError as exc:
                raise RuntimeError(f"{market}{code} name is not valid GBK") from exc
            if not name:
                raise RuntimeError(f"{market}{code} has a price but no security name")
            rows.append({
                "trade_date": f"{ymd[:4]}-{ymd[4:6]}-{ymd[6:]}",
                "market": market, "symbol": code, "name": name,
                "prev_close": round(prev_close, 4), "open": round(open_, 4),
                "high": round(high, 4), "low": round(low, 4),
                "close": round(close, 4), "volume": int(volume),
                "amount": round(amount, 2),
            })
        if len(rows) - before < int(thresholds[market]):
            raise RuntimeError(
                f"{market} priced rows {len(rows)-before} below minimum "
                f"{thresholds[market]}; refusing partial market package"
            )
    if not rows:
        raise RuntimeError("TongDaXin package contains no priced securities")
    frame = pd.DataFrame(rows)
    if frame.duplicated(["trade_date", "market", "symbol"]).any():
        raise RuntimeError("TongDaXin package contains duplicate security/date rows")
    return rows


def normalize_tdx_package(
    rows: list[dict], *, available_time=None, source_url: str = TDX_PACKAGE_URL
) -> pd.DataFrame:
    """Attach immutable provenance and explicitly mark retrieval-time-only data."""
    if not rows:
        raise ValueError("rows cannot be empty")
    captured = _capture_time(available_time)
    frame = pd.DataFrame(rows).copy()
    required = {"trade_date", "market", "symbol", "name", "prev_close", "open",
                "high", "low", "close", "volume", "amount"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError("TDX rows missing columns: " + ",".join(sorted(missing)))
    if frame.duplicated(["trade_date", "market", "symbol"]).any():
        raise ValueError("duplicate TDX symbol/date rows")
    frame["available_time"] = captured
    frame["source_id"] = "tdx.official_daily_package"
    frame["source_url"] = source_url
    frame["availability_basis"] = "retrieval_time_only"
    frame["historical_pit_eligible"] = False
    frame["scoring_eligible"] = False
    revisions = []
    for row in frame.to_dict("records"):
        payload = {key: row[key] for key in sorted(required)}
        revisions.append(hashlib.sha256(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest())
    frame["source_revision"] = revisions
    return frame


def fetch_tdx_daily_package(
    trade_date, *, available_time=None, get=requests.get, timeout=(10, 90),
    min_priced=None,
) -> pd.DataFrame:
    """Fetch one full-market SSE/SZSE/BSE daily package; fail closed on gaps."""
    ymd = _trade_day(trade_date)
    url = TDX_PACKAGE_URL.format(ymd=ymd)
    try:
        response = get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=timeout)
    except requests.RequestException as exc:
        raise RuntimeError(f"TongDaXin package request failed: {exc}") from exc
    if response.status_code == 404:
        raise ValueError(
            f"{ymd}: package unavailable (non-trading day, not yet published, "
            "or outside publisher retention)"
        )
    try:
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(f"TongDaXin package HTTP error: {exc}") from exc
    rows = _parse_package(response.content, ymd, min_priced=min_priced)
    return normalize_tdx_package(rows, available_time=available_time, source_url=url)


def audit_tdx_snapshot(frame: pd.DataFrame) -> dict:
    """Produce exact coverage counts; no guessed universe denominator."""
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError("TDX snapshot missing columns: " + ",".join(sorted(missing)))
    if frame.empty:
        raise ValueError("TDX snapshot is empty")
    if frame.duplicated(["trade_date", "market", "symbol"]).any():
        raise ValueError("TDX snapshot contains duplicate securities")
    if frame["historical_pit_eligible"].astype(bool).any():
        raise ValueError("retrieval-time TDX data cannot be historical PIT eligible")
    markets = {}
    for market in ("sh", "sz", "bj"):
        part = frame.loc[frame.market == market]
        markets[market] = {
            "priced_rows": int(len(part)),
            "unique_symbols": int(part.symbol.nunique()),
            "amount_sum_cny": round(float(part.amount.sum()), 2),
        }
    return {
        "trade_date": str(frame.trade_date.iloc[0]),
        "available_time": _capture_time(frame.available_time.iloc[0]).isoformat(),
        "total_priced_rows": int(len(frame)),
        "unique_symbols": int(frame[["market", "symbol"]].drop_duplicates().shape[0]),
        "markets": markets,
        "source_id": "tdx.official_daily_package",
        "historical_pit_eligible": False,
        "scoring_eligible": False,
        "status": "validated_snapshot_not_historical_pit",
    }
