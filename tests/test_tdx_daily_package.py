import io
import struct
import zipfile

import pandas as pd
import pytest

from src.providers.tdx_daily_package import (
    _parse_package, audit_tdx_snapshot, normalize_tdx_package,
)


def _package(ymd="20261113", counts=(2, 2, 2)):
    data = io.BytesIO()
    with zipfile.ZipFile(data, "w", zipfile.ZIP_DEFLATED) as archive:
        for market, count in zip(("sh", "sz", "bj"), counts):
            codes = bytearray()
            quotes = bytearray()
            for index in range(count):
                code = f"{index + 1:06d}".encode("ascii")
                record = bytearray(150)
                record[:6] = code
                struct.pack_into("<H", record, 32, index)
                record[40:40 + len("测试".encode("gbk"))] = "测试".encode("gbk")
                codes.extend(record)
                block = bytearray(512)
                struct.pack_into("<d", block, 4, 9.5)
                struct.pack_into("<4d", block, 12, 10.0, 11.0, 9.0, 10.5)
                struct.pack_into("<Q", block, 56, 100000)
                struct.pack_into("<d", block, 72, 1000000.0)
                quotes.extend(block)
            archive.writestr(f"{market}{ymd[2:]}.cod", bytes(codes))
            archive.writestr(f"{market}{ymd[2:]}.md1", bytes(quotes))
    return data.getvalue()


def test_parse_full_market_package_and_all_three_markets():
    rows = _parse_package(
        _package(), "20261113", min_priced={"sh": 1, "sz": 1, "bj": 1}
    )
    assert len(rows) == 6
    assert {row["market"] for row in rows} == {"sh", "sz", "bj"}
    assert all(row["volume"] == 100000 for row in rows)
    assert all(row["amount"] == 1000000.0 for row in rows)


def test_normalization_marks_backfilled_history_not_pit_eligible():
    rows = _parse_package(
        _package(), "20261113", min_priced={"sh": 1, "sz": 1, "bj": 1}
    )
    frame = normalize_tdx_package(rows, available_time="2026-11-14T08:00:00+08:00")
    assert frame["historical_pit_eligible"].eq(False).all()
    assert frame["availability_basis"].eq("retrieval_time_only").all()
    assert frame["available_time"].nunique() == 1
    assert frame["source_revision"].str.len().eq(64).all()


def test_audit_reports_exact_counts_and_amount_by_market():
    rows = _parse_package(
        _package(), "20261113", min_priced={"sh": 1, "sz": 1, "bj": 1}
    )
    frame = normalize_tdx_package(rows, available_time="2026-11-14T00:00:00Z")
    audit = audit_tdx_snapshot(frame)
    assert audit["total_priced_rows"] == 6
    assert audit["unique_symbols"] == 2
    assert audit["markets"]["sh"]["priced_rows"] == 2
    assert audit["markets"]["bj"]["amount_sum_cny"] == 2000000.0
    assert audit["historical_pit_eligible"] is False


def test_duplicate_security_rows_fail_closed():
    row = {
        "trade_date": "2026-11-13", "market": "sh", "symbol": "600000",
        "name": "样例", "prev_close": 9.5, "open": 10, "high": 11,
        "low": 9, "close": 10.5, "volume": 100, "amount": 1000,
    }
    with pytest.raises(ValueError, match="duplicate"):
        normalize_tdx_package([row, row], available_time="2026-11-14T00:00:00Z")


def test_naive_availability_timestamp_is_rejected():
    rows = _parse_package(
        _package(), "20261113", min_priced={"sh": 1, "sz": 1, "bj": 1}
    )
    with pytest.raises(ValueError, match="timezone-aware"):
        normalize_tdx_package(rows, available_time="2026-11-14 00:00:00")


def test_malformed_package_fails_closed():
    with pytest.raises(RuntimeError, match="not a ZIP"):
        _parse_package(b"bad response", "20261113")
