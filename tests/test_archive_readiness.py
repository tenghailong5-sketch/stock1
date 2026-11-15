from __future__ import annotations

import json

from src.model.archive_readiness import audit_forward_archive


def test_archive_readiness_never_upgrades_retrieval_time_rows(tmp_path):
    dataset = tmp_path / "stock_daily"
    dataset.mkdir()
    path = dataset / "000001.jsonl"
    rows = [
        {
            "symbol": "000001", "trade_date": "2026-01-05",
            "available_time": "2026-11-12T08:00:00Z", "source_id": "akshare",
            "source_revision": "v1", "availability_basis": "retrieval_time_only",
            "historical_pit_eligible": False,
        },
        {
            "symbol": "000002", "trade_date": "2026-01-05",
            "available_time": "2026-01-05T08:00:00Z", "source_id": "exchange",
            "source_revision": "v2", "availability_basis": "original_publication_time",
            "historical_pit_eligible": True,
        },
        {"symbol": "000003", "trade_date": "2026-01-05"},
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    result = audit_forward_archive(tmp_path)
    file_result = result["files"][0]
    assert result["status"] == "PIT_BLOCKED"
    assert file_result["row_count"] == 3
    assert file_result["pit_eligible_count"] == 1
    assert file_result["retrieval_only_count"] == 1
    assert file_result["missing_lineage_count"] == 1
    assert file_result["status"] == "MIXED_PIT_AND_BLOCKED"


def test_archive_readiness_reports_empty_and_missing_roots(tmp_path):
    assert audit_forward_archive(tmp_path / "missing")["status"] == "MISSING_ARCHIVE_ROOT"
    (tmp_path / "empty").mkdir()
    (tmp_path / "empty" / "blank.jsonl").write_text("", encoding="utf-8")
    result = audit_forward_archive(tmp_path / "empty")
    assert result["files"][0]["status"] == "EMPTY"
    assert result["datasets"][0]["status"] == "EMPTY"
