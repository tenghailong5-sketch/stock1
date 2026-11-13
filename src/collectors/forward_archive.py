"""Forward-only daily archive collector for index and industry snapshots.

This collector records retrieval-time observations. It does not claim historical
point-in-time availability for previously published market data.
"""
from __future__ import annotations

import argparse
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
    normalize_index_history_snapshot,
    normalize_market_breadth_snapshot,
)
from src.providers.kaipanhong_market import fetch_historical_market_context

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


def collect(ak_module, archive_root: Path, *, captured_at=None, write: bool = True, retry_sleep=time.sleep, industry_clock=None, observation_clock=None, allow_partial: bool = False, historical_market_fetcher=None) -> dict[str, int]:
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

    eastmoney_industry = None
    eastmoney_error = None
    try:
        eastmoney_industry = fetch_current_industry_membership_snapshot(
            ak_module, clock=industry_clock, pause_seconds=0.15
        )
    except Exception as exc:
        eastmoney_error = exc

    def _coverage(snapshot):
        if snapshot is None or snapshot.empty or not expected_symbols:
            return 0.0
        members = set(snapshot["symbol"].astype(str).str.zfill(6))
        return len(members & expected_symbols) / len(expected_symbols)

    industry = eastmoney_industry
    eastmoney_coverage = _coverage(eastmoney_industry)
    # Trigger the independent Sina taxonomy whenever Eastmoney is unavailable
    # or covers less than 95% of the current quote universe. Keep the better
    # snapshot; never combine incompatible taxonomies into a fake unified map.
    if eastmoney_industry is None or eastmoney_coverage < 0.95:
        try:
            sina_industry = fetch_sina_current_industry_membership_snapshot(
                ak_module, clock=industry_clock
            )
            sina_coverage = _coverage(sina_industry)
            if industry is None or sina_coverage > eastmoney_coverage:
                industry = sina_industry
                chosen_source = "akshare.stock_sector_detail_sina"
                chosen_coverage = sina_coverage
            else:
                chosen_source = "akshare.stock_board_industry_cons_em"
                chosen_coverage = eastmoney_coverage
            if chosen_coverage >= 0.95:
                source_status["industry_membership"] = (
                    f"ok: {chosen_source}; symbol_coverage={chosen_coverage:.1%}"
                )
            else:
                source_status["industry_membership"] = (
                    f"partial: {chosen_source}; symbol_coverage={chosen_coverage:.1%}; "
                    f"Eastmoney={eastmoney_coverage:.1%}; Sina={sina_coverage:.1%}; "
                    f"Eastmoney_error={type(eastmoney_error).__name__ if eastmoney_error else 'none'}:"
                    f"{str(eastmoney_error)[:180] if eastmoney_error else 'no error'}"
                )
                if not allow_partial:
                    raise RuntimeError(source_status["industry_membership"])
        except Exception as sina_exc:
            if industry is None:
                source_status["industry_membership"] = (
                    "unavailable: Eastmoney "
                    f"{type(eastmoney_error).__name__ if eastmoney_error else 'unknown'}: "
                    f"{str(eastmoney_error)[:180] if eastmoney_error else 'no snapshot'} | "
                    f"Sina {type(sina_exc).__name__}: {str(sina_exc)[:120]}"
                )
                if not allow_partial:
                    raise RuntimeError(source_status["industry_membership"]) from sina_exc
            else:
                source_status["industry_membership"] = (
                    f"partial: Eastmoney symbol_coverage={eastmoney_coverage:.1%}; "
                    f"Sina fallback failed: {type(sina_exc).__name__}: {str(sina_exc)[:120]}"
                )
                if not allow_partial:
                    raise RuntimeError(source_status["industry_membership"]) from sina_exc
    else:
        source_status["industry_membership"] = (
            f"ok: akshare.stock_board_industry_cons_em; symbol_coverage={eastmoney_coverage:.1%}"
        )

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
        report = {
            "captured_at": captured.isoformat(),
            "source_status": source_status,
            "rows_appended": counts,
            "partial": any(not status.startswith("ok") for status in source_status.values()),
            "note": "Unavailable sources are explicitly recorded; missing data is not imputed.",
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

    counts = collect(
        ak, Path(args.output), write=not args.dry_run, allow_partial=True,
        historical_market_fetcher=fetch_historical_market_context,
    )
    print(json.dumps({"mode": "dry_run" if args.dry_run else "archive", "validated_rows": counts}, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
