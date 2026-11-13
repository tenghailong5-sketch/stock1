import pandas as pd

from scripts.probe_tencent_universe import attach_row_lineage


def test_tencent_probe_attaches_retrieval_time_and_explicit_non_pit_status():
    frame = pd.DataFrame({
        "date": [pd.Timestamp("2026-11-12")],
        "open": [10.0], "close": [10.2], "high": [10.3], "low": [9.9],
        "volume": [1000], "amount": [10200.0],
    })

    result = attach_row_lineage(frame, "sh600519", "2026-11-13T07:00:00+00:00")

    assert result.loc[0, "symbol"] == "sh600519"
    assert result.loc[0, "available_time"] == "2026-11-13T07:00:00+00:00"
    assert result.loc[0, "source_id"] == "akshare.stock_zh_a_hist_tx"
    assert len(result.loc[0, "source_revision"]) == 64
    assert result.loc[0, "historical_pit_eligible"] == False
    assert result.loc[0, "availability_basis"] == "retrieval_time_only"


def test_tencent_probe_row_revision_changes_when_provider_values_change():
    frame = pd.DataFrame({
        "date": [pd.Timestamp("2026-11-12")],
        "open": [10.0], "close": [10.2], "high": [10.3], "low": [9.9],
        "volume": [1000], "amount": [10200.0],
    })
    original = attach_row_lineage(frame, "sh600519", "2026-11-13T07:00:00+00:00")
    revised = frame.copy()
    revised.loc[0, "close"] = 10.25
    changed = attach_row_lineage(revised, "sh600519", "2026-11-13T07:01:00+00:00")

    assert original.loc[0, "source_revision"] != changed.loc[0, "source_revision"]
