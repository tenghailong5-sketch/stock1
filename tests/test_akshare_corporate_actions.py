from pathlib import Path

import pandas as pd
import pytest

from src.providers.akshare_corporate_actions import (
    archive_corporate_action_snapshot,
    archive_shortlist_corporate_actions,
    normalize_corporate_actions,
)


def test_dividend_snapshot_is_forward_only_and_preserves_first_seen():
    raw = pd.DataFrame([{
        "公告日期": "2026-05-01", "送股": 2, "转增": 3, "派息": 1.5,
        "进度": "实施", "除权除息日": "2026-05-20", "股权登记日": "2026-05-19",
        "红股上市日": "2026-05-20",
    }])
    first = normalize_corporate_actions(
        raw, symbol="600000", indicator="分红",
        observed_at="2026-11-15T00:00:00Z",
    ).iloc[0]
    assert first["first_seen_at"] == "2026-11-15T00:00:00+00:00"
    assert first["available_time"] == first["first_seen_at"]
    assert not bool(first["historical_pit_eligible"])
    assert first["eligible_for_decisions_after_first_seen"] is True
    assert first["corporate_action_status"] == "CAPTURED"

    changed = raw.copy()
    changed.loc[0, "进度"] = "实施完成"
    revised = normalize_corporate_actions(
        changed, symbol="600000", indicator="分红",
        observed_at="2026-11-16T00:00:00Z",
        first_seen_by_key={first["event_key"]: first["first_seen_at"]},
    ).iloc[0]
    assert revised["first_seen_at"] == first["first_seen_at"]
    assert revised["observed_at"] == "2026-11-16T00:00:00+00:00"
    assert revised["source_revision"] != first["source_revision"]


def test_rights_issue_fields_are_normalized():
    raw = pd.DataFrame([{
        "公告日期": "2025-03-01", "配股方案(每10股配股股数)": "3.0",
        "配股价格(元)": "5.20", "基准股本(股)": "1000000",
        "除权日": "2025-04-01", "股权登记日": "2025-03-31",
        "缴款起始日": "2025-04-02", "缴款终止日": "2025-04-08",
        "配股上市日": "2025-04-15", "募集资金合计(元)": "1500000",
    }])
    row = normalize_corporate_actions(
        raw, symbol="000001", indicator="配股",
        observed_at="2026-11-15T00:00:00Z",
    ).iloc[0]
    assert row["event_type"] == "rights_issue"
    assert row["rights_issue_ratio"] == 3.0
    assert row["rights_issue_price"] == 5.2
    assert row["base_share_capital"] == 1000000.0
    assert not bool(row["historical_pit_eligible"])


def test_invalid_time_and_missing_announcement_fail_closed():
    with pytest.raises(ValueError, match="timezone-aware"):
        normalize_corporate_actions(
            pd.DataFrame([{"公告日期": "2025-01-01"}]),
            symbol="000001", indicator="分红", observed_at="2026-11-15",
        )
    with pytest.raises(ValueError, match="announcement_date"):
        normalize_corporate_actions(
            pd.DataFrame([{"公告日期": "--", "派息": 1}]),
            symbol="000001", indicator="分红",
            observed_at="2026-11-15T00:00:00Z",
        )


def test_archive_appends_revisions_without_backdating_first_seen(tmp_path):
    class FakeAk:
        def __init__(self):
            self.progress = "实施"

        def stock_history_dividend_detail(self, *, symbol, indicator):
            if indicator == "分红":
                return pd.DataFrame([{
                    "公告日期": "2026-01-01", "送股": 0, "转增": 0, "派息": 1.0,
                    "进度": self.progress, "除权除息日": "2026-02-01",
                    "股权登记日": "2026-01-31", "红股上市日": "--",
                }])
            return pd.DataFrame(columns=["公告日期", "配股方案(每10股配股股数)"])

    api = FakeAk()
    root = Path(tmp_path)
    first = archive_corporate_action_snapshot(
        "600000", ak_module=api, observed_at="2026-11-15T00:00:00Z",
        archive_root=root,
    )
    api.progress = "实施完成"
    second = archive_corporate_action_snapshot(
        "600000", ak_module=api, observed_at="2026-11-16T00:00:00Z",
        archive_root=root,
    )
    lines = (root / "corporate_actions" / "600000.jsonl").read_text().splitlines()
    records = [__import__("json").loads(line) for line in lines]
    assert first["rows_appended"] == 1
    assert second["rows_appended"] == 1
    assert len(records) == 2
    assert {row["first_seen_at"] for row in records} == {"2026-11-15T00:00:00+00:00"}
    assert len({row["source_revision"] for row in records}) == 2


def test_shortlist_is_bounded_and_does_not_scan_full_universe(tmp_path):
    class FakeAk:
        def __init__(self):
            self.calls = []

        def stock_history_dividend_detail(self, *, symbol, indicator):
            self.calls.append((symbol, indicator))
            if indicator == "分红":
                return pd.DataFrame([{"公告日期": "2026-01-01", "派息": 1.0}])
            return pd.DataFrame(columns=["公告日期"])

    api = FakeAk()
    candidates = pd.DataFrame({
        "symbol": [f"{i:06d}" for i in range(1, 31)],
        "快速初筛分": list(range(30)),
    })
    audit = archive_shortlist_corporate_actions(
        candidates, ak_module=api, observed_at="2026-11-15T00:00:00Z",
        archive_root=tmp_path, max_symbols=4,
    )
    assert audit["symbols_checked"] == 4
    assert len(api.calls) == 8
    assert not bool(audit["historical_pit_eligible"])
