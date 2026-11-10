"""Forward-only daily archive collector for index and industry snapshots.

This collector records retrieval-time observations. It does not claim historical
point-in-time availability for previously published market data.
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from src.providers.akshare_limit_pools import fetch_limit_pool_snapshots
from src.providers.akshare_pit import (
    fetch_current_industry_membership_snapshot,
    normalize_index_history_snapshot,
    normalize_market_breadth_snapshot,
)

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


def collect(ak_module, archive_root: Path, *, captured_at=None, write: bool = True) -> dict[str, int]:
    captured = pd.Timestamp(
        captured_at if captured_at is not None else datetime.now(ZoneInfo("Asia/Shanghai"))
    )
    if captured.tzinfo is None:
        raise ValueError("captured_at must be timezone-aware")
    captured = captured.tz_convert("UTC")
    start_date = (captured - pd.Timedelta(days=180)).strftime("%Y%m%d")
    end_date = captured.strftime("%Y%m%d")

    # Fetch and validate every source before writing anything. A provider error
    # therefore cannot leave a partially refreshed set of archives.
    index_records: list[dict] = []
    for code in INDEXES:
        raw = ak_module.index_zh_a_hist(
            symbol=code, period="daily", start_date=start_date, end_date=end_date
        )
        normalized = normalize_index_history_snapshot(
            raw, symbol=code, fetched_at=captured
        )
        index_records.extend(_records(normalized))

    industry = fetch_current_industry_membership_snapshot(
        ak_module, clock=lambda: captured, pause_seconds=0.15
    )
    industry_records = _records(industry)
    breadth = normalize_market_breadth_snapshot(
        ak_module.stock_zh_a_spot_em(), fetched_at=captured
    )
    breadth_records = _records(breadth)
    limit_pools = fetch_limit_pool_snapshots(
        ak_module,
        trade_date=captured.tz_convert("Asia/Shanghai").strftime("%Y%m%d"),
        captured_at=captured,
    )
    limit_pool_records = _records(limit_pools)

    if not write:
        return {
            **{f"index_{code}": sum(1 for row in index_records if row["index_symbol"] == code) for code in INDEXES},
            "industry_membership": len(industry_records),
            "market_breadth": len(breadth_records),
            "limit_pools": len(limit_pool_records),
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
        ("symbol", "industry_code", "source_revision"),
    )
    counts["market_breadth"] = append_jsonl_atomic(
        archive_root / "market_breadth" / f"{day}.jsonl",
        breadth_records,
        ("snapshot_time", "source_revision"),
    )
    counts["limit_pools"] = append_jsonl_atomic(
        archive_root / "limit_pools" / f"{day}.jsonl",
        limit_pool_records,
        ("event_type", "trade_date", "source_revision"),
    )
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture forward-only A-share market data")
    parser.add_argument("--output", default="data/forward_archive")
    parser.add_argument("--dry-run", action="store_true", help="fetch and validate all sources without writing archive files")
    args = parser.parse_args()
    import akshare as ak

    counts = collect(ak, Path(args.output), write=not args.dry_run)
    print(json.dumps({"mode": "dry_run" if args.dry_run else "archive", "validated_rows": counts}, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
