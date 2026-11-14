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



def test_summarize_sw_index_history_reports_dates_and_never_claims_pit():
    from src.collectors.forward_archive import _summarize_sw_index_history

    summary = _summarize_sw_index_history({"801010": [
        {"trade_date": "2024-01-02", "historical_pit_eligible": False,
         "availability_basis": "retrieval_time_only"},
        {"trade_date": "2024-01-03", "historical_pit_eligible": False,
         "availability_basis": "retrieval_time_only"},
        {"trade_date": "2024-01-03", "historical_pit_eligible": False,
         "availability_basis": "retrieval_time_only"},
    ]})
    assert summary["801010"]["row_count"] == 3
    assert summary["801010"]["unique_trade_date_count"] == 2
    assert summary["801010"]["first_trade_date"] == "2024-01-02"
    assert summary["801010"]["last_trade_date"] == "2024-01-03"
    assert summary["801010"]["historical_pit_eligible_row_count"] == 0
    assert summary["801010"]["availability_basis_values"] == ["retrieval_time_only"]
