import pandas as pd

from src.collectors.forward_archive import _collect_sw_index_history


def test_collect_sw_index_history_archives_each_active_industry_without_pit_promotion():
    snapshot = pd.DataFrame({"industry_code": ["801010", "801020", "801010"]})

    def fetcher(code, *, clock=None):
        return pd.DataFrame([{
            "trade_date": pd.Timestamp("2026-11-13"),
            "industry_code": code,
            "close": 1234.5,
            "available_time": pd.Timestamp("2026-11-14T00:00:00Z"),
            "availability_basis": "retrieval_time_only",
            "historical_pit_eligible": False,
            "source_revision": "a" * 64,
        }])

    records, failures = _collect_sw_index_history(snapshot, fetcher)
    assert sorted(records) == ["801010", "801020"]
    assert sum(map(len, records.values())) == 2
    assert failures == {}
    assert all(row["historical_pit_eligible"] is False
               for rows in records.values() for row in rows)


def test_collect_sw_index_history_records_failures_per_code_and_continues():
    snapshot = pd.DataFrame({"industry_code": ["801010", "801020"]})

    def fetcher(code, *, clock=None):
        if code == "801010":
            raise RuntimeError("provider unavailable")
        return pd.DataFrame([{
            "trade_date": pd.Timestamp("2026-11-13"),
            "industry_code": code,
            "close": 1234.5,
            "available_time": pd.Timestamp("2026-11-14T00:00:00Z"),
            "availability_basis": "retrieval_time_only",
            "historical_pit_eligible": False,
            "source_revision": "b" * 64,
        }])

    records, failures = _collect_sw_index_history(snapshot, fetcher)
    assert sorted(records) == ["801020"]
    assert "801010" in failures
