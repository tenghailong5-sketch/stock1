"""Audit forward archives without upgrading retrieval-time history to PIT data."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


_ALLOWED_PIT_BASES = {"original_publication_time", "exchange_publication_time"}
_LINEAGE_FIELDS = ("available_time", "source_id", "source_revision",
                   "availability_basis", "historical_pit_eligible")


def _truthy_pit(value) -> bool:
    return value is True or (not pd.isna(value) and str(value).strip().lower() in {"true", "1", "yes"})


def _audit_file(path: Path) -> dict:
    total = eligible = retrieval_only = missing_lineage = invalid_time = 0
    symbols: set[str] = set()
    dates: set[str] = set()
    sources: set[str] = set()
    errors: list[str] = []
    if not path.exists() or path.stat().st_size == 0:
        return {
            "dataset": path.parent.name, "file": str(path), "row_count": 0,
            "pit_eligible_count": 0, "retrieval_only_count": 0,
            "missing_lineage_count": 0, "invalid_available_time_count": 0,
            "symbol_count": 0, "trade_date_count": 0, "sources": [],
            "status": "EMPTY", "errors": [],
        }
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            total += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                errors.append(f"invalid_json_line:{line_no}")
                continue
            for field in ("symbol", "code"):
                value = row.get(field)
                if value is not None and str(value).strip():
                    symbols.add(str(value).strip())
            if row.get("trade_date") is not None:
                dates.add(str(row["trade_date"])[:10])
            if row.get("source_id"):
                sources.add(str(row["source_id"]).strip())
            absent = [key for key in _LINEAGE_FIELDS if key not in row or row[key] is None or str(row[key]).strip() == ""]
            if absent:
                missing_lineage += 1
                continue
            try:
                timestamp = pd.Timestamp(row["available_time"])
                if timestamp.tzinfo is None:
                    raise ValueError("naive timestamp")
            except (TypeError, ValueError, OverflowError):
                invalid_time += 1
                continue
            basis = str(row["availability_basis"]).strip()
            if basis == "retrieval_time_only" or not _truthy_pit(row["historical_pit_eligible"]):
                retrieval_only += 1
                continue
            if basis in _ALLOWED_PIT_BASES:
                eligible += 1
            else:
                retrieval_only += 1
    if errors:
        status = "CORRUPT"
    elif total == 0:
        status = "EMPTY"
    elif eligible == total:
        status = "PIT_READY"
    elif eligible:
        status = "MIXED_PIT_AND_BLOCKED"
    else:
        status = "PIT_BLOCKED"
    return {
        "dataset": path.parent.name, "file": str(path), "row_count": total,
        "pit_eligible_count": eligible, "retrieval_only_count": retrieval_only,
        "missing_lineage_count": missing_lineage, "invalid_available_time_count": invalid_time,
        "symbol_count": len(symbols), "trade_date_count": len(dates),
        "sources": sorted(sources), "status": status, "errors": errors,
    }


def audit_forward_archive(archive_root: str | Path = "data/forward_archive") -> dict:
    """Return per-file and per-dataset PIT readiness; never infer historical eligibility."""
    root = Path(archive_root)
    if not root.exists():
        return {"archive_root": str(root), "status": "MISSING_ARCHIVE_ROOT",
                "files": [], "datasets": []}
    files = sorted(path for path in root.rglob("*.jsonl") if path.is_file())
    file_rows = [_audit_file(path) for path in files]
    grouped: dict[str, list[dict]] = {}
    for item in file_rows:
        grouped.setdefault(item["dataset"], []).append(item)
    datasets = []
    for name, rows in sorted(grouped.items()):
        total = sum(row["row_count"] for row in rows)
        eligible = sum(row["pit_eligible_count"] for row in rows)
        status = "EMPTY" if total == 0 else "PIT_READY" if eligible == total else "PIT_BLOCKED"
        datasets.append({
            "dataset": name, "file_count": len(rows), "row_count": total,
            "pit_eligible_count": eligible,
            "retrieval_only_count": sum(row["retrieval_only_count"] for row in rows),
            "missing_lineage_count": sum(row["missing_lineage_count"] for row in rows),
            "invalid_available_time_count": sum(row["invalid_available_time_count"] for row in rows),
            "symbol_count": len(set().union(*(set(row["sources"]) for row in rows))),
            "status": status,
        })
    return {
        "archive_root": str(root),
        "status": "PIT_READY" if datasets and all(x["status"] in {"PIT_READY", "EMPTY"} for x in datasets) else "PIT_BLOCKED",
        "file_count": len(file_rows), "datasets": datasets, "files": file_rows,
    }
