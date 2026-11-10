import json

import pandas as pd
import pytest

from src.collectors.forward_archive import append_jsonl_atomic, collect


class FakeArchiveAkshare:
    def index_zh_a_hist(self, symbol, period, start_date, end_date):
        return pd.DataFrame([{
            "日期": "2026-11-10", "开盘": 3000, "最高": 3050, "最低": 2990,
            "收盘": 3030, "成交量": 100000, "成交额": 200000000,
        }])

    def stock_zh_a_spot_em(self):
        return pd.DataFrame([
            {"代码": "000001", "最新价": 10, "涨跌幅": 2.0},
            {"代码": "600000", "最新价": 8, "涨跌幅": -1.0},
            {"代码": "000002", "最新价": 9, "涨跌幅": 0.0},
        ])


    def stock_zt_pool_em(self, date):
        return pd.DataFrame([{"代码": "000001", "名称": "样例", "连板数": 2}])

    def stock_zt_pool_dtgc_em(self, date):
        return pd.DataFrame(columns=["代码", "名称"])

    def stock_zt_pool_zbgc_em(self, date):
        return pd.DataFrame([{"代码": "600000", "名称": "样例炸板", "炸板次数": 1}])

    def stock_zt_pool_previous_em(self, date):
        return pd.DataFrame([{"代码": "000002", "名称": "昨日涨停"}])

    def stock_board_industry_name_em(self):
        return pd.DataFrame([
            {"板块名称": "电子", "板块代码": "BK0001"},
            {"板块名称": "计算机", "板块代码": "BK0002"},
        ])

    def stock_board_industry_cons_em(self, symbol):
        return pd.DataFrame({"代码": ["000001", "600000"]})


def test_append_jsonl_is_idempotent_and_preserves_existing_rows(tmp_path):
    path = tmp_path / "archive.jsonl"
    first = {"symbol": "000001", "trade_date": "2026-11-10", "revision": "a"}
    second = {"symbol": "000002", "trade_date": "2026-11-10", "revision": "b"}
    assert append_jsonl_atomic(path, [first], ("symbol", "trade_date", "revision")) == 1
    assert append_jsonl_atomic(
        path, [first, second], ("symbol", "trade_date", "revision")
    ) == 1
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert rows == [first, second]


def test_corrupt_archive_is_rejected_without_overwriting(tmp_path):
    path = tmp_path / "archive.jsonl"
    path.write_text("{broken\n", encoding="utf-8")
    with pytest.raises(ValueError, match="corrupt archive"):
        append_jsonl_atomic(path, [{"id": 1}], ("id",))
    assert path.read_text(encoding="utf-8") == "{broken\n"


def test_collector_archives_indexes_and_daily_industry_snapshot(tmp_path):
    counts = collect(
        FakeArchiveAkshare(),
        tmp_path,
        captured_at="2026-11-11T16:30:00+08:00",
    )
    assert set(counts) == {
        "index_000001", "index_399001", "index_399006",
        "index_000688", "index_000300", "industry_membership", "market_breadth", "limit_pools",
    }
    assert all(counts[f"index_{code}"] == 1 for code in ("000001", "399001", "399006", "000688", "000300"))
    assert counts["industry_membership"] == 4  # two symbols across two industries
    assert counts["market_breadth"] == 1
    index_file = tmp_path / "index_daily" / "000001.jsonl"
    index_row = json.loads(index_file.read_text(encoding="utf-8").splitlines()[0])
    assert index_row["historical_pit_eligible"] is False
    assert index_row["availability_basis"] == "retrieval_time_only"
    assert index_row["available_time"] == "2026-11-11T08:30:00.000Z"

    industry_file = tmp_path / "industry_membership" / "2026-11-11.jsonl"
    industry_rows = [json.loads(line) for line in industry_file.read_text(encoding="utf-8").splitlines()]
    assert len(industry_rows) == 4
    assert {row["snapshot_scope"] for row in industry_rows} == {
        "current_membership_observation_only"
    }
    assert all("valid_from" not in row and "valid_to" not in row for row in industry_rows)

    breadth_file = tmp_path / "market_breadth" / "2026-11-11.jsonl"
    breadth_row = json.loads(breadth_file.read_text(encoding="utf-8").splitlines()[0])
    assert breadth_row["advances"] == 1
    assert breadth_row["declines"] == 1
    assert breadth_row["unchanged"] == 1
    assert breadth_row["snapshot_scope"] == "current_provider_snapshot_only"
    assert breadth_row["historical_pit_eligible"] is False


def test_collector_refuses_naive_capture_time(tmp_path):
    with pytest.raises(ValueError, match="timezone-aware"):
        collect(FakeArchiveAkshare(), tmp_path, captured_at="2026-11-11T08:30:00")
