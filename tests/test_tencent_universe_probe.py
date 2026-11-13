import pandas as pd
from types import SimpleNamespace

from scripts.probe_tencent_universe import (
    attach_row_lineage,
    fetch_corporate_action_audit,
    stratified_sample,
)


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

def test_tencent_probe_selects_fixed_exchange_proportional_100_symbol_sample():
    universe = pd.DataFrame({
        "symbol": (
            [f"sh{600000+i:06d}" for i in range(150)]
            + [f"sz{1+i:06d}" for i in range(300)]
            + [f"bj{430000+i:06d}" for i in range(50)]
        )
    })
    picks = stratified_sample(universe, sample_size=100)

    assert len(picks) == 100
    assert len(set(picks)) == 100
    assert sum(x.startswith("sh") for x in picks) == 30
    assert sum(x.startswith("sz") for x in picks) == 60
    assert sum(x.startswith("bj") for x in picks) == 10


def test_corporate_action_audit_accepts_empty_schema_and_marks_non_pit():
    fake = SimpleNamespace(
        stock_dividend_cninfo=lambda symbol: pd.DataFrame(columns=[
            "实施方案公告日期", "送股比例", "转增比例", "派息比例", "除权日"
        ]),
        stock_allotment_cninfo=lambda symbol, start_date, end_date: pd.DataFrame(columns=[
            "公告日期", "配股比例", "除权日"
        ]),
    )

    statuses, frames = fetch_corporate_action_audit(fake, "sh600519", sleep_fn=lambda _: None)

    assert [item["status"] for item in statuses] == ["ok_response", "ok_response"]
    assert [item["row_count"] for item in statuses] == [0, 0]
    assert all(item["historical_pit_eligible"] is False for item in statuses)
    assert frames == []
