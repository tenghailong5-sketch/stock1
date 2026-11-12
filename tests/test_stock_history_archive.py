import json

import pandas as pd
import pytest

from src.fhm100 import archive_stock_history_snapshot


def _bars(close=10.0):
    return pd.DataFrame([
        {"日期": "2026-11-10", "开盘": 9.8, "最高": 10.2, "最低": 9.7,
         "收盘": close, "成交量": 100000, "成交额": 1000000, "换手率": 1.2},
        {"日期": "2026-11-11", "开盘": 10.0, "最高": 10.5, "最低": 9.9,
         "收盘": 10.3, "成交量": 120000, "成交额": 1300000, "换手率": 1.4},
    ])


def test_stock_history_archive_records_retrieval_time_and_pit_ineligibility(tmp_path):
    result = archive_stock_history_snapshot(
        _bars(), "000001", captured_at="2026-11-12T08:30:00+00:00",
        archive_root=tmp_path,
    )
    assert result == {"rows_appended": 2, "manifests_appended": 1}

    rows_path = tmp_path / "stock_daily" / "2026-11-12.jsonl"
    rows = [json.loads(line) for line in rows_path.read_text().splitlines()]
    assert len(rows) == 2
    assert {row["symbol"] for row in rows} == {"000001"}
    assert {row["available_time"] for row in rows} == {"2026-11-12T08:30:00.000Z"}
    assert all(row["availability_basis"] == "retrieval_time_only" for row in rows)
    assert all(row["historical_pit_eligible"] is False for row in rows)
    assert all(row["snapshot_scope"] == "current_symbol_history_retrieval" for row in rows)
    assert len({row["snapshot_id"] for row in rows}) == 1
    assert {row["trade_date"][:10] for row in rows} == {"2026-11-10", "2026-11-11"}

    manifest_path = tmp_path / "stock_daily_manifest" / "2026-11-12.jsonl"
    manifests = [json.loads(line) for line in manifest_path.read_text().splitlines()]
    assert len(manifests) == 1
    assert manifests[0]["row_count"] == 2
    assert manifests[0]["historical_pit_eligible"] is False
    assert manifests[0]["availability_basis"] == "retrieval_time_only"


def test_stock_history_archive_is_idempotent_for_unchanged_bars_but_keeps_capture_manifest(tmp_path):
    first = archive_stock_history_snapshot(
        _bars(), "000001", captured_at="2026-11-12T08:30:00+00:00",
        archive_root=tmp_path,
    )
    second = archive_stock_history_snapshot(
        _bars(), "000001", captured_at="2026-11-12T08:35:00+00:00",
        archive_root=tmp_path,
    )
    assert first["rows_appended"] == 2
    assert second["rows_appended"] == 0
    assert second["manifests_appended"] == 1

    rows_path = tmp_path / "stock_daily" / "2026-11-12.jsonl"
    assert len(rows_path.read_text().splitlines()) == 2
    manifest_path = tmp_path / "stock_daily_manifest" / "2026-11-12.jsonl"
    assert len(manifest_path.read_text().splitlines()) == 2


def test_stock_history_archive_preserves_provider_revisions(tmp_path):
    archive_stock_history_snapshot(
        _bars(), "000001", captured_at="2026-11-12T08:30:00+00:00",
        archive_root=tmp_path,
    )
    archive_stock_history_snapshot(
        _bars(close=10.1), "000001", captured_at="2026-11-12T08:40:00+00:00",
        archive_root=tmp_path,
    )
    rows_path = tmp_path / "stock_daily" / "2026-11-12.jsonl"
    rows = [json.loads(line) for line in rows_path.read_text().splitlines()]
    revised = [row for row in rows if row["trade_date"].startswith("2026-11-10")]
    assert len(revised) == 2
    assert len({row["source_revision"] for row in revised}) == 2
    assert {row["available_time"] for row in revised} == {
        "2026-11-12T08:30:00.000Z", "2026-11-12T08:40:00.000Z"
    }


def test_stock_history_archive_rejects_naive_capture_time(tmp_path):
    with pytest.raises(ValueError, match="timezone-aware"):
        archive_stock_history_snapshot(
            _bars(), "000001", captured_at="2026-11-12T08:30:00",
            archive_root=tmp_path,
        )
