"""Forward-only daily archive collector for index and industry snapshots.

This collector records retrieval-time observations. It does not claim historical
point-in-time availability for previously published market data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import time

import requests
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from src.model.market_sentiment import summarize_limit_pool_sentiment
from src.providers.akshare_limit_pools import fetch_limit_pool_snapshots
from src.providers.akshare_pit import (
    fetch_current_industry_membership_snapshot,
    fetch_sina_current_industry_membership_snapshot,
    fetch_eastmoney_individual_industry_membership_snapshot,
    normalize_index_history_snapshot,
    normalize_market_breadth_snapshot,
)
from src.providers.kaipanhong_market import fetch_historical_market_context
from src.providers.baostock_industry import fetch_baostock_industry_membership_snapshot
from src.providers.sw_industry import fetch_sw_current_industry_membership_snapshot
from src.providers.sw_api_industry import fetch_sw_api_industry_membership_snapshot
from src.providers.baidu_industry import fetch_baidu_industry_membership_snapshot
from src.providers.cninfo_industry import fetch_cninfo_current_industry_membership_snapshot

def _call_with_network_retry(call, *args, attempts: int = 3, sleep_fn=time.sleep, **kwargs):
    """Retry transient transport failures only; never mask schema/data errors."""
    if attempts < 1:
        raise ValueError("attempts must be at least 1")
    for attempt in range(attempts):
        try:
            return call(*args, **kwargs)
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
            if attempt + 1 >= attempts:
                raise
            sleep_fn(2 ** attempt)


class _RetryingProvider:
    """Transparent AKShare proxy that retries transient HTTP transport failures."""
    def __init__(self, provider, *, attempts: int = 3, sleep_fn=time.sleep):
        self._provider = provider
        self._attempts = attempts
        self._sleep_fn = sleep_fn

    def __getattr__(self, name):
        attribute = getattr(self._provider, name)
        if not callable(attribute):
            return attribute
        def invoke(*args, **kwargs):
            return _call_with_network_retry(
                attribute, *args, attempts=self._attempts,
                sleep_fn=self._sleep_fn, **kwargs
            )
        return invoke


INDEXES = {
    "000001": "SSE Composite",
    "399001": "SZSE Component",
    "399006": "ChiNext",
    "000688": "STAR 50",
    "000300": "CSI 300",
}


def _record_key(record: dict, fields: tuple[str, ...]) -> str:
    return json.dumps([record.get(field) for field in fields], ensure_ascii=False)


def append_jsonl_atomic(path: Path, records: list[dict], key_fields: tuple[str, ...]) -> int:
    """Append unseen records atomically; existing archive content is never truncated."""
    path.parent.mkdir(parents=True, exist_ok=True)
    existing_keys: set[str] = set()
    old_bytes = b""
    if path.exists():
        old_bytes = path.read_bytes()
        for line_number, line in enumerate(old_bytes.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"corrupt archive {path} at line {line_number}") from exc
            existing_keys.add(_record_key(item, key_fields))

    additions = []
    for record in records:
        key = _record_key(record, key_fields)
        if key in existing_keys:
            continue
        existing_keys.add(key)
        additions.append(json.dumps(record, ensure_ascii=False, sort_keys=True, default=str))

    if not additions:
        return 0

    payload = old_bytes
    if payload and not payload.endswith(b"\n"):
        payload += b"\n"
    payload += ("\n".join(additions) + "\n").encode("utf-8")
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    return len(additions)



def _load_manual_industry_evidence(
    path: Path, expected_symbols: set[str], *, captured_at
) -> pd.DataFrame:
    """Load source-linked, current-only manual observations into the audit union.

    This loader never creates selected-taxonomy model rows or historical PIT data.
    Only reconciled Sina/10jqka F10 records with a real HTTPS source URL qualify.
    """
    columns = [
        "symbol", "industry_code", "industry_name", "snapshot_id",
        "expected_industry_count", "observed_at", "available_time",
        "availability_basis", "historical_pit_eligible", "source_id",
        "snapshot_scope", "source_revision", "classification_system",
        "source_url", "evidence_status",
    ]
    if not path.exists():
        return pd.DataFrame(columns=columns)
    captured = pd.Timestamp(captured_at)
    if captured.tzinfo is None:
        raise ValueError("manual evidence capture time must be timezone-aware")
    captured = captured.tz_convert("UTC")
    rows = []
    allowed_hosts = {"money.finance.sina.com.cn", "basic.10jqka.com.cn"}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid manual industry evidence at line {line_number}") from exc
        if item.get("record_status") != "source_url_reconciled_current_observation_candidate":
            continue
        symbol = str(item.get("symbol", "")).zfill(6)
        source_url = str(item.get("source_url", "")).strip()
        source_id = str(item.get("source_id", "")).strip()
        taxonomy = str(item.get("classification_system", "")).strip()
        industry_name = str(item.get("industry_name", "")).strip()
        if symbol not in expected_symbols:
            continue
        if not (source_url.startswith("https://") and any(
            source_url.startswith("https://" + host + "/") for host in allowed_hosts
        )):
            continue
        if source_id not in {"sina.finance_f10_sw", "10jqka.f10_sw"}:
            continue
        if not taxonomy.startswith("申万行业"):
            continue
        if not industry_name:
            continue
        revision_payload = "|".join((symbol, industry_name, taxonomy, source_url))
        revision = hashlib.sha256(revision_payload.encode("utf-8")).hexdigest()
        rows.append({
            "symbol": symbol,
            "industry_code": industry_name,
            "industry_name": industry_name,
            "snapshot_id": f"manual-web-{captured.strftime('%Y%m%dT%H%M%S')}",
            "expected_industry_count": 0,
            "observed_at": captured,
            "available_time": captured,
            "availability_basis": "retrieval_time_only",
            "historical_pit_eligible": False,
            "source_id": source_id,
            "snapshot_scope": "current_membership_observation_only",
            "source_revision": revision,
            "classification_system": taxonomy,
            "source_url": source_url,
            "evidence_status": "source_url_reconciled_current_observation",
        })
    return pd.DataFrame(rows, columns=columns)


def _records(frame: pd.DataFrame) -> list[dict]:
    if frame.empty:
        raise ValueError("provider returned an empty snapshot; refusing to archive it")
    payload = frame.to_json(orient="records", date_format="iso", force_ascii=False)
    return json.loads(payload)


def _tencent_breadth_frame(ak_module):
    """Convert Tencent's independent spot snapshot into the normalized AKShare shape."""
    provider = getattr(ak_module, "stock_zh_a_spot_tx", None)
    if not callable(provider):
        raise RuntimeError("Tencent spot fallback is unavailable")
    frame = provider()
    required = {"code", "zxj", "zdf", "turnover"}
    if frame is None or frame.empty or required - set(frame.columns):
        missing = sorted(required - set(frame.columns)) if frame is not None else sorted(required)
        raise RuntimeError("Tencent spot fallback schema changed: " + ",".join(missing))
    out = pd.DataFrame({
        "代码": frame["code"].astype(str).str.strip().str.lower().str[-6:].str.zfill(6),
        "最新价": pd.to_numeric(frame["zxj"], errors="coerce"),
        "涨跌幅": pd.to_numeric(frame["zdf"], errors="coerce"),
        "成交额": pd.to_numeric(frame["turnover"], errors="coerce") * 10000,
    })
    if out.empty or out["代码"].duplicated().any():
        raise RuntimeError("Tencent spot fallback is empty or contains duplicate symbols")
    return out


def collect(ak_module, archive_root: Path, *, captured_at=None, write: bool = True, retry_sleep=time.sleep, industry_clock=None, observation_clock=None, allow_partial: bool = False, historical_market_fetcher=None, baostock_module=None, sw_industry_fetcher=None, baidu_industry_fetcher=None, sw_api_industry_fetcher=None, cninfo_industry_fetcher=None) -> dict[str, int]:
    ak_module = _RetryingProvider(ak_module, sleep_fn=retry_sleep)
    captured = pd.Timestamp(
        captured_at if captured_at is not None else datetime.now(ZoneInfo("Asia/Shanghai"))
    )
    if captured.tzinfo is None:
        raise ValueError("captured_at must be timezone-aware")
    captured = captured.tz_convert("UTC")
    start_date = (captured - pd.Timedelta(days=180)).strftime("%Y%m%d")
    end_date = captured.strftime("%Y%m%d")
    observed_now = observation_clock or (lambda: datetime.now(timezone.utc))

    # Fetch and validate every source before writing anything. A provider error
    # therefore cannot leave a partially refreshed set of archives.
    source_status = {}
    index_records: list[dict] = []
    for code in INDEXES:
        index_source = "akshare.index_zh_a_hist"
        try:
            raw = ak_module.index_zh_a_hist(
                symbol=code, period="daily", start_date=start_date, end_date=end_date
            )
        except Exception as exc:
            # Eastmoney Push2 can temporarily block a runner IP. Switch provider
            # once to Tencent; never retry the blocked Eastmoney endpoint.
            if type(exc).__name__ != "Push2RateLimitError":
                raise
            if not hasattr(ak_module, "stock_zh_index_daily_tx"):
                raise RuntimeError(
                    f"Eastmoney index provider rate-limited and Tencent fallback is unavailable for {code}"
                ) from exc
            prefix = "sh" if code.startswith("0") else "sz"
            fallback = ak_module.stock_zh_index_daily_tx(
                symbol=prefix + code, start_date=start_date, end_date=end_date
            )
            if fallback is None or fallback.empty:
                raise RuntimeError(f"Tencent index fallback returned empty data for {code}") from exc
            required = {"date", "open", "high", "low", "close", "volume"}
            if required - set(fallback.columns):
                raise RuntimeError(
                    f"Tencent index fallback schema changed for {code}: "
                    + ",".join(sorted(required - set(fallback.columns)))
                ) from exc
            raw_data = {
                "日期": fallback["date"],
                "开盘": fallback["open"],
                "最高": fallback["high"],
                "最低": fallback["low"],
                "收盘": fallback["close"],
                "成交量": fallback["volume"],
            }
            # Current AKShare exposes Tencent turnover as amount (yuan).
            # Older releases exposed amount as volume (lots); do not guess that
            # legacy schema or silently relabel turnover as volume.
            if "amount" in fallback.columns:
                raw_data["成交额"] = fallback["amount"]
            raw = pd.DataFrame(raw_data)
            index_source = "tencent.stock_zh_index_daily_tx"
        # Availability is stamped after this specific provider response returns,
        # not at batch start; otherwise a slow multi-index collection backdates PIT.
        index_observed_at = pd.Timestamp(observed_now())
        if index_observed_at.tzinfo is None:
            raise ValueError("observation_clock must return timezone-aware timestamps")
        normalized = normalize_index_history_snapshot(
            raw, symbol=code, fetched_at=index_observed_at
        )
        rows = _records(normalized)
        index_records.extend(rows)
        source_status[f"index_{code}"] = (
            f"ok: {index_source}; rows={len(rows)}"
        )

    # Collect the current quote universe before industry membership so we can
    # measure actual symbol coverage. A successful provider response is not
    # sufficient evidence that its constituent list covers the market.
    breadth_raw = None
    breadth_records: list[dict] = []
    try:
        try:
            breadth_raw = ak_module.stock_zh_a_spot_em()
            breadth_source = "akshare.stock_zh_a_spot_em"
        except Exception as exc:
            if type(exc).__name__ != "Push2RateLimitError":
                raise
            breadth_raw = _tencent_breadth_frame(ak_module)
            breadth_source = "tencent.stock_zh_a_spot_tx"
        breadth_observed_at = pd.Timestamp(observed_now())
        if breadth_observed_at.tzinfo is None:
            raise ValueError("observation_clock must return timezone-aware timestamps")
        breadth = normalize_market_breadth_snapshot(
            breadth_raw, fetched_at=breadth_observed_at, source_id=breadth_source
        )
        breadth_records = _records(breadth)
        source_status["market_breadth"] = f"ok: {breadth_source}"
    except Exception as exc:
        source_status["market_breadth"] = f"unavailable: {type(exc).__name__}: {str(exc)[:240]}"
        if not allow_partial:
            raise


    industry_records: list[dict] = []
    expected_symbols: set[str] = set()
    if breadth_raw is not None and not breadth_raw.empty:
        if "代码" in breadth_raw.columns:
            raw_symbols = breadth_raw["代码"]
        elif "code" in breadth_raw.columns:
            raw_symbols = breadth_raw["code"].astype(str).str.lower().str[-6:]
        else:
            raw_symbols = pd.Series(dtype=str)
        expected_symbols = set(
            raw_symbols.astype(str).str.extract(r"(\d{6})", expand=False).dropna().str.zfill(6)
        )

    industry_source_snapshots: list[pd.DataFrame] = []
    eastmoney_industry = None
    eastmoney_error = None
    try:
        eastmoney_industry = fetch_current_industry_membership_snapshot(
            ak_module, clock=industry_clock, pause_seconds=1.0
        )
    except Exception as exc:
        eastmoney_error = exc

    if eastmoney_industry is not None and not eastmoney_industry.empty:
        industry_source_snapshots.append(eastmoney_industry.copy())

    # Coverage denominator is the full current quote universe; READY means exactly 100%.
    def _coverage(snapshot):
        if snapshot is None or snapshot.empty or not expected_symbols:
            return 0.0
        members = set(snapshot["symbol"].astype(str).str.zfill(6))
        return len(members & expected_symbols) / len(expected_symbols)

    industry = eastmoney_industry
    eastmoney_coverage = _coverage(eastmoney_industry)
    chosen_source = "akshare.stock_board_industry_cons_em"
    chosen_coverage = eastmoney_coverage
    sina_coverage = None
    sina_secondary_coverage = None
    sina_secondary_error = None
    individual_coverage = None
    individual_failures = 0
    baostock_coverage = None
    sina_error = None
    individual_error = None
    baostock_error = None
    sw_coverage = None
    sw_error = None
    sw_api_coverage = None
    sw_api_error = None
    baidu_coverage = None
    baidu_error = None

    # Always query independent taxonomies for cross-source audit and per-symbol gap coverage.
    # The raw observations are archived separately; never mix their labels in the
    # legacy single-taxonomy model-consumption snapshot.
    if expected_symbols:
        try:
            sina_industry = fetch_sina_current_industry_membership_snapshot(
                ak_module, clock=industry_clock
            )
            sina_coverage = _coverage(sina_industry)
            if sina_industry is not None and not sina_industry.empty:
                industry_source_snapshots.append(sina_industry.copy())
            if sina_coverage > chosen_coverage:
                industry = sina_industry
                chosen_source = "akshare.stock_sector_detail_sina"
                chosen_coverage = sina_coverage
        except Exception as exc:
            sina_error = exc

        # Sina's separate "行业" directory is a second independent taxonomy,
        # not a synonym for "新浪行业". Archive it under its own source_id.
        try:
            sina_secondary = fetch_sina_current_industry_membership_snapshot(
                ak_module, clock=industry_clock, indicator="行业"
            )
            sina_secondary_coverage = _coverage(sina_secondary)
            if sina_secondary is not None and not sina_secondary.empty:
                industry_source_snapshots.append(sina_secondary.copy())
        except Exception as exc:
            sina_secondary_error = exc

    # Baostock is an independent free source; use it only when it improves
    # coverage. Its updateDate is metadata, not a historical publication time.
    # Inject the provider from the production CLI; tests never contact its socket.
    if expected_symbols and baostock_module is not None:
        try:
            baostock_industry = fetch_baostock_industry_membership_snapshot(
                baostock_module, clock=industry_clock
            )
            baostock_coverage = _coverage(baostock_industry)
            if baostock_industry is not None and not baostock_industry.empty:
                industry_source_snapshots.append(baostock_industry.copy())
            if baostock_coverage > chosen_coverage:
                industry = baostock_industry
                chosen_source = "baostock.query_stock_industry"
                chosen_coverage = baostock_coverage
        except Exception as exc:
            baostock_error = exc
    elif expected_symbols:
        baostock_error = RuntimeError("Baostock provider was not injected into this collector run")

    # SW Research current index API: an independent live membership endpoint.
    # It is called even when earlier sources are complete so the raw source layer
    # can retain independent observations and expose disagreements.
    if expected_symbols and sw_api_industry_fetcher is not None:
        try:
            sw_api_industry = sw_api_industry_fetcher(clock=industry_clock)
            sw_api_coverage = _coverage(sw_api_industry)
            if sw_api_industry is not None and not sw_api_industry.empty:
                industry_source_snapshots.append(sw_api_industry.copy())
            if sw_api_coverage > chosen_coverage:
                industry = sw_api_industry
                chosen_source = "swsresearch.index_publish.component_stocks"
                chosen_coverage = sw_api_coverage
        except Exception as exc:
            sw_api_error = exc
    elif expected_symbols:
        sw_api_error = RuntimeError("SW API provider was not injected into this collector run")

    # Shenwan Hongyuan publishes a bulk SW2021 stock-classification workbook.
    # It avoids per-symbol requests and is independent of Eastmoney/Baostock.
    if expected_symbols and sw_industry_fetcher is not None:
        try:
            sw_industry = sw_industry_fetcher(clock=industry_clock)
            sw_coverage = _coverage(sw_industry)
            if sw_industry is not None and not sw_industry.empty:
                industry_source_snapshots.append(sw_industry.copy())
            if sw_coverage > chosen_coverage:
                industry = sw_industry
                chosen_source = "swsresearch.sw2021_stock_classification"
                chosen_coverage = sw_coverage
        except Exception as exc:
            sw_error = exc
    elif expected_symbols:
        sw_error = RuntimeError("Shenwan provider was not injected into this collector run")


    # CNINFO CSRC classification is an independent per-security source.
    # Query only symbols still missing from the multi-source union to limit load.
    cninfo_coverage = None
    cninfo_error = None
    cninfo_failed_symbols = 0
    if expected_symbols and cninfo_industry_fetcher is not None:
        try:
            covered_so_far = set().union(*[
                set(snap["symbol"].astype(str).str.zfill(6)) & expected_symbols
                for snap in industry_source_snapshots if snap is not None and not snap.empty
            ]) if industry_source_snapshots else set()
            cninfo_missing = sorted(expected_symbols - covered_so_far)
            cninfo_industry = cninfo_industry_fetcher(
                ak_module, cninfo_missing, clock=industry_clock,
                pause_seconds=0.5,
            )
            cninfo_coverage = _coverage(cninfo_industry)
            cninfo_failed_symbols = len(getattr(cninfo_industry, "attrs", {}).get("failed_symbols", {}))
            if cninfo_industry is not None and not cninfo_industry.empty:
                industry_source_snapshots.append(cninfo_industry.copy())
        except Exception as exc:
            cninfo_error = exc
    elif expected_symbols:
        cninfo_error = RuntimeError("CNINFO provider was not injected into this collector run")


    # Only use per-security fallbacks after every bulk source has had a chance.
    # Restrict Eastmoney individual lookups to symbols still missing from the
    # cross-source union, and never let a lower-coverage taxonomy replace a
    # better selected snapshot.
    if expected_symbols and eastmoney_coverage < 1.0:
        rate_limit_text = (
            f"{type(eastmoney_error).__name__ if eastmoney_error else ''} "
            f"{str(eastmoney_error) if eastmoney_error else ''}"
        ).lower()
        eastmoney_rate_limited = any(
            marker in rate_limit_text
            for marker in (
                "ratelimit", "rate limit", "cooldown", "too many requests",
                "请求过于频繁", "访问过于频繁", "40分钟", "40 minutes",
                "5分钟", "5 minutes",
            )
        )
        if eastmoney_rate_limited:
            individual_error = RuntimeError("skipped_due_to_eastmoney_rate_limit")
        elif eastmoney_industry is None or eastmoney_industry.empty:
            individual_error = RuntimeError("skipped_due_to_missing_eastmoney_board_snapshot")
        else:
            try:
                board_members = set(
                    eastmoney_industry["symbol"].astype(str).str.zfill(6)
                )
                covered_before_individual = set().union(*[
                    set(snapshot["symbol"].astype(str).str.zfill(6)) & expected_symbols
                    for snapshot in industry_source_snapshots
                    if snapshot is not None and not snapshot.empty
                ]) if industry_source_snapshots else set()
                missing_symbols = expected_symbols - covered_before_individual
                industry_code_by_name = (
                    eastmoney_industry.drop_duplicates("industry_name")
                    .set_index("industry_name")["industry_code"].astype(str).to_dict()
                )
                if missing_symbols:
                    individual_industry = fetch_eastmoney_individual_industry_membership_snapshot(
                        ak_module, missing_symbols, clock=industry_clock,
                        pause_seconds=1.0, industry_code_by_name=industry_code_by_name,
                    )
                    individual_failures = len(individual_industry.attrs.get("failed_symbols", {}))
                    if not individual_industry.empty:
                        industry_source_snapshots.append(individual_industry.copy())
                        individual_industry["snapshot_id"] = eastmoney_industry["snapshot_id"].iloc[0]
                        combined_eastmoney = pd.concat(
                            [eastmoney_industry, individual_industry], ignore_index=True
                        )
                        industry_count = int(combined_eastmoney["industry_code"].nunique())
                        combined_eastmoney["expected_industry_count"] = industry_count
                        individual_coverage = _coverage(combined_eastmoney)
                        if individual_coverage > chosen_coverage:
                            industry = combined_eastmoney
                            chosen_source = "eastmoney_board_plus_individual_info"
                            chosen_coverage = individual_coverage
                    else:
                        individual_coverage = eastmoney_coverage
                else:
                    individual_coverage = None
            except Exception as exc:
                individual_error = exc

    # Baidu Gushitong provides an independent per-security industry lookup.
    # Query the full verified universe in bounded batches; preserve Baidu's own
    # taxonomy and select it only when its measured symbol coverage is better.
    baidu_requested_symbols = sorted(
        expected_symbols - (
            set().union(*[
                set(frame["symbol"].astype(str).str.zfill(6))
                for frame in industry_source_snapshots
                if frame is not None and not frame.empty
            ]) if industry_source_snapshots else set()
        )
    )
    if baidu_requested_symbols and baidu_industry_fetcher is not None:
        try:
            baidu_industry = baidu_industry_fetcher(baidu_requested_symbols)
            baidu_coverage = _coverage(baidu_industry)
            if baidu_industry is not None and not baidu_industry.empty:
                industry_source_snapshots.append(baidu_industry.copy())
            if baidu_coverage > chosen_coverage:
                industry = baidu_industry
                chosen_source = "baidu.gushitong.relatedblock"
                chosen_coverage = baidu_coverage
        except Exception as exc:
            baidu_error = exc
    elif baidu_requested_symbols:
        baidu_error = RuntimeError("quarantined: Baidu PAE getrelatedblock disabled pending live endpoint validation")

    # Reconciled manual F10 evidence fills individual current-observation gaps.
    # It is archived only in the multi-source audit layer, never in the selected
    # single-taxonomy snapshot consumed by the model.
    manual_evidence_path = (
        archive_root / "industry_manual_verified"
        / f"{captured.tz_convert('Asia/Shanghai').strftime('%Y-%m-%d')}.jsonl"
    )
    manual_evidence = _load_manual_industry_evidence(
        manual_evidence_path, expected_symbols, captured_at=observed_now()
    )
    if not manual_evidence.empty:
        industry_source_snapshots.append(manual_evidence)
        source_status["manual_industry_evidence"] = (
            f"ok: source_url_reconciled={len(manual_evidence)}; current-only; PIT-ineligible"
        )
    else:
        source_status["manual_industry_evidence"] = "ok: no eligible source-linked manual evidence"

    source_coverage = {
        "eastmoney_board": eastmoney_coverage,
        "eastmoney_individual": individual_coverage,
        "sina": sina_coverage,
        "sina_secondary_industry": sina_secondary_coverage,
        "cninfo_csrc": cninfo_coverage,
        "baostock": baostock_coverage,
        "shenwan_xls": sw_coverage,
        "shenwan_api": sw_api_coverage,
        "baidu": baidu_coverage,
    }
    source_symbols = {}
    for snapshot in industry_source_snapshots:
        source_id = str(snapshot["source_id"].iloc[0]) if "source_id" in snapshot.columns and not snapshot.empty else "unknown"
        source_symbols.setdefault(source_id, set()).update(
            set(snapshot["symbol"].astype(str).str.zfill(6)) & expected_symbols
        )
    multi_source_covered_symbols = set().union(*source_symbols.values()) if source_symbols else set()
    uncovered_symbols = sorted(expected_symbols - multi_source_covered_symbols)
    multi_source_coverage = (
        len(multi_source_covered_symbols) / len(expected_symbols) if expected_symbols else 0.0
    )
    source_status["industry_membership_multisource"] = (
        f"ok: covered={len(multi_source_covered_symbols)}/{len(expected_symbols)} "
        if multi_source_coverage >= 1.0 else
        f"partial: covered={len(multi_source_covered_symbols)}/{len(expected_symbols)}; "
        f"uncovered={len(uncovered_symbols)}"
    )
    industry_source_records = (
        _records(pd.concat(industry_source_snapshots, ignore_index=True, sort=False))
        if industry_source_snapshots else []
    )

    if industry is None or industry.empty:
        source_status["industry_membership"] = (
            "unavailable: "
            f"Eastmoney={type(eastmoney_error).__name__ if eastmoney_error else 'no_snapshot'}: "
            f"{str(eastmoney_error)[:120] if eastmoney_error else 'no data'}; "
            f"individual={type(individual_error).__name__ if individual_error else 'not_available'}: "
            f"{str(individual_error)[:120] if individual_error else 'no data'}; "
            f"Sina={type(sina_error).__name__ if sina_error else 'not_attempted'}: "
            f"{str(sina_error)[:100] if sina_error else 'no data'}; "
            f"Baostock={type(baostock_error).__name__ if baostock_error else 'not_attempted'}: "
            f"{str(baostock_error)[:100] if baostock_error else 'no data'}; "
            f"Shenwan={type(sw_error).__name__ if sw_error else 'not_attempted'}: "
            f"{str(sw_error)[:100] if sw_error else 'no data'}"
        )
    elif chosen_coverage >= 1.0 and chosen_source == "akshare.stock_sector_detail_sina":
        source_status["industry_membership"] = (
            f"ok: {chosen_source}; symbol_coverage={chosen_coverage:.1%}"
        )
    else:
        prefix = "ok" if chosen_coverage >= 1.0 else "partial"
        source_status["industry_membership"] = (
            f"{prefix}: {chosen_source}; symbol_coverage={chosen_coverage:.1%}; "
            f"Eastmoney_board={eastmoney_coverage:.1%}; "
            f"Eastmoney_individual={f'{individual_coverage:.1%}' if individual_coverage is not None else 'not_available'}; "
            f"individual_failures={individual_failures}; "
            f"Sina={f'{sina_coverage:.1%}' if sina_coverage is not None else 'not_available'}; "
            f"Baostock={f'{baostock_coverage:.1%}' if baostock_coverage is not None else 'not_available'}; "
            f"Shenwan_XLS={f'{sw_coverage:.1%}' if sw_coverage is not None else 'not_available'}; "
            f"Shenwan_API={f'{sw_api_coverage:.1%}' if sw_api_coverage is not None else 'not_available'}"
        )
    if not allow_partial and chosen_coverage < 1.0:
        raise RuntimeError(source_status["industry_membership"])

    if industry is not None and not industry.empty:
        industry_records = _records(industry)
    limit_pool_records: list[dict] = []
    sentiment = None
    try:
        limit_pools = fetch_limit_pool_snapshots(
            ak_module,
            trade_date=captured.tz_convert("Asia/Shanghai").strftime("%Y%m%d"),
            captured_at=None,
            clock=observed_now,
        )
        limit_pool_records = _records(limit_pools)
        sentiment = summarize_limit_pool_sentiment(limit_pools)
        sentiment["trade_date"] = captured.tz_convert("Asia/Shanghai").strftime("%Y-%m-%d")
        sentiment["capture_time"] = pd.to_datetime(limit_pools["capture_time"], utc=True).max().isoformat()
        source_status["limit_pools"] = "ok"
    except Exception as exc:
        source_status["limit_pools"] = f"unavailable: {type(exc).__name__}: {str(exc)[:240]}"
        if not allow_partial:
            raise

    historical_breadth_records: list[dict] = []
    historical_sentiment_records: list[dict] = []
    historical_market_failures: list[str] = []
    if historical_market_fetcher is not None and expected_symbols:
        index_dates = sorted({
            pd.Timestamp(row.get("trade_date")).strftime("%Y-%m-%d")
            for row in index_records
            if row.get("trade_date") is not None
        })[-119:]
        current_day = captured.tz_convert("Asia/Shanghai").strftime("%Y-%m-%d")
        for history_day in index_dates:
            # Today's current breadth and limit pools are collected above.
            if history_day >= current_day:
                continue
            breadth_path = archive_root / "market_breadth" / f"{history_day}.jsonl"
            sentiment_path = archive_root / "market_sentiment" / f"{history_day}.jsonl"

            def _has_kph_source(path):
                if not path.exists() or path.stat().st_size == 0:
                    return False
                try:
                    rows = [
                        json.loads(line)
                        for line in path.read_text(encoding="utf-8").splitlines()
                        if line.strip()
                    ]
                except (OSError, json.JSONDecodeError):
                    return False
                return any(
                    row.get("source_id") == "kaipanhong.market_emotion_kph"
                    for row in rows
                )

            if _has_kph_source(breadth_path) and _has_kph_source(sentiment_path):
                continue
            try:
                historical_breadth, historical_sentiment = historical_market_fetcher(
                    history_day, expected_market_count=len(expected_symbols)
                )
                historical_breadth_records.append(historical_breadth)
                historical_sentiment_records.append(historical_sentiment)
            except Exception as exc:
                historical_market_failures.append(
                    f"{history_day}:{type(exc).__name__}:{str(exc)[:140]}"
                )
        fetched_days = len({
            row["trade_date"] for row in historical_breadth_records
        } | {
            row["trade_date"] for row in historical_sentiment_records
        })
        if historical_market_failures:
            source_status["historical_market_context"] = (
                f"partial: kaipanhong.market_emotion_kph; "
                f"backfilled={fetched_days}; failed={len(historical_market_failures)}; "
                + " | ".join(historical_market_failures[:3])
            )
        else:
            source_status["historical_market_context"] = (
                f"ok: kaipanhong.market_emotion_kph; backfilled={fetched_days}"
            )
    else:
        source_status["historical_market_context"] = (
            "ok: historical backfill not requested or no verified current market universe"
        )

    if not write:
        return {
            **{f"index_{code}": sum(1 for row in index_records if row["index_symbol"] == code) for code in INDEXES},
            "industry_membership": len(industry_records),
            "industry_membership_sources": len(industry_source_records),
            "market_breadth": len(breadth_records),
            "limit_pools": len(limit_pool_records),
            "market_sentiment": int(sentiment is not None),
            "market_breadth_history_backfill": len(historical_breadth_records),
            "market_sentiment_history_backfill": len(historical_sentiment_records),
        }

    counts: dict[str, int] = {}
    for code in INDEXES:
        rows = [row for row in index_records if row["index_symbol"] == code]
        counts[f"index_{code}"] = append_jsonl_atomic(
            archive_root / "index_daily" / f"{code}.jsonl",
            rows,
            ("index_symbol", "trade_date", "source_revision"),
        )

    day = captured.tz_convert("Asia/Shanghai").strftime("%Y-%m-%d")
    counts["industry_membership"] = append_jsonl_atomic(
        archive_root / "industry_membership" / f"{day}.jsonl",
        industry_records,
        ("snapshot_id", "symbol", "industry_code", "source_revision"),
    ) if industry_records else 0
    counts["industry_membership_sources"] = append_jsonl_atomic(
        archive_root / "industry_membership_sources" / f"{day}.jsonl",
        industry_source_records,
        ("snapshot_id", "source_id", "symbol", "industry_code", "source_revision"),
    ) if industry_source_records else 0
    counts["market_breadth"] = append_jsonl_atomic(
        archive_root / "market_breadth" / f"{day}.jsonl",
        breadth_records,
        ("snapshot_time", "source_revision"),
    ) if breadth_records else 0
    counts["limit_pools"] = append_jsonl_atomic(
        archive_root / "limit_pools" / f"{day}.jsonl",
        limit_pool_records,
        ("event_type", "trade_date", "source_revision"),
    ) if limit_pool_records else 0
    counts["market_sentiment"] = append_jsonl_atomic(
        archive_root / "market_sentiment" / f"{day}.jsonl",
        [sentiment],
        ("trade_date", "capture_time"),
    ) if sentiment is not None else 0
    counts["market_breadth_history_backfill"] = 0
    for record in historical_breadth_records:
        history_day = str(record["trade_date"])
        counts["market_breadth_history_backfill"] += append_jsonl_atomic(
            archive_root / "market_breadth" / f"{history_day}.jsonl",
            [record],
            ("trade_date", "source_id", "source_revision"),
        )
    counts["market_sentiment_history_backfill"] = 0
    for record in historical_sentiment_records:
        history_day = str(record["trade_date"])
        counts["market_sentiment_history_backfill"] += append_jsonl_atomic(
            archive_root / "market_sentiment" / f"{history_day}.jsonl",
            [record],
            ("trade_date", "source_id", "source_revision"),
        )
    if allow_partial:
        def _error_detail(error):
            if error is None:
                return None
            return f"{type(error).__name__}: {str(error)[:500]}"

        eastmoney_symbols = (
            set(eastmoney_industry["symbol"].astype(str).str.zfill(6))
            if eastmoney_industry is not None and not eastmoney_industry.empty else set()
        )
        selected_symbols = (
            set(industry["symbol"].astype(str).str.zfill(6))
            if industry is not None and not industry.empty else set()
        )
        report = {
            "captured_at": captured.isoformat(),
            "source_status": source_status,
            "industry_audit": {
                "quote_universe_symbol_count": len(expected_symbols),
                "selected_snapshot_symbol_count": len(selected_symbols),
                "selected_snapshot_covered_symbol_count": len(selected_symbols & expected_symbols),
                "selected_snapshot_coverage": chosen_coverage,
                "eastmoney_board_symbol_count": len(eastmoney_symbols),
                "eastmoney_board_coverage": eastmoney_coverage,
                "eastmoney_board_error": _error_detail(eastmoney_error),
                "eastmoney_individual_coverage": individual_coverage,
                "eastmoney_individual_error": _error_detail(individual_error),
                "eastmoney_individual_failed_symbol_count": individual_failures,
                "sina_coverage": sina_coverage,
                "sina_error": _error_detail(sina_error),
                "sina_secondary_industry_coverage": sina_secondary_coverage,
                "sina_secondary_industry_error": _error_detail(sina_secondary_error),
                "cninfo_coverage": cninfo_coverage,
                "cninfo_error": _error_detail(cninfo_error),
                "cninfo_failed_symbol_count": cninfo_failed_symbols,
                "baostock_coverage": baostock_coverage,
                "baostock_error": _error_detail(baostock_error),
                "shenwan_coverage": sw_coverage,
                "shenwan_error": _error_detail(sw_error),
                "shenwan_api_coverage": sw_api_coverage,
                "shenwan_api_error": _error_detail(sw_api_error),
                "shenwan_api_failed_industry_count": (
                    len(getattr(sw_api_industry, "attrs", {}).get("failed_industry_codes", []))
                    if "sw_api_industry" in locals() and sw_api_industry is not None else None
                ),
                "shenwan_api_failed_industry_codes": (
                    getattr(sw_api_industry, "attrs", {}).get("failed_industry_codes", [])
                    if "sw_api_industry" in locals() and sw_api_industry is not None else []
                ),
                "baidu_coverage": baidu_coverage,
                "baidu_error": _error_detail(baidu_error),
                "source_coverage": source_coverage,
                "multi_source_covered_symbol_count": len(multi_source_covered_symbols),
                "multi_source_coverage": multi_source_coverage,
                "uncovered_symbols": uncovered_symbols,
                "source_record_count": len(industry_source_records),
                "source_industry_code_pair_count": len({
                    (str(row.get("source_id")), str(row.get("industry_code")))
                    for row in industry_source_records
                }),
                "baidu_requested_symbol_count": len(baidu_requested_symbols),
                "selected_source": chosen_source,
            },
            "rows_appended": counts,
            "partial": any(not status.startswith("ok") for status in source_status.values()),
            "note": "Coverage denominator is the distinct symbol count from this run's captured quote universe. Provider failures include exception details; missing data is not imputed.",
        }
        report_path = archive_root / "collection_report" / f"{day}.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture forward-only A-share market data")
    parser.add_argument("--output", default="data/forward_archive")
    parser.add_argument("--dry-run", action="store_true", help="fetch and validate all sources without writing archive files")
    args = parser.parse_args()
    import akshare as ak
    import baostock as bs

    counts = collect(
        ak, Path(args.output), write=not args.dry_run, allow_partial=True,
        historical_market_fetcher=fetch_historical_market_context,
        baostock_module=bs,
        sw_industry_fetcher=fetch_sw_current_industry_membership_snapshot,
        sw_api_industry_fetcher=fetch_sw_api_industry_membership_snapshot,
        cninfo_industry_fetcher=fetch_cninfo_current_industry_membership_snapshot,
    )
    print(json.dumps({"mode": "dry_run" if args.dry_run else "archive", "validated_rows": counts}, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
