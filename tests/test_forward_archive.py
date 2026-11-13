import json

import pandas as pd
import pytest

from src.collectors.forward_archive import append_jsonl_atomic, collect, _call_with_network_retry
from types import SimpleNamespace

from src.providers.baostock_industry import fetch_baostock_industry_membership_snapshot
from src.providers.akshare_pit import (
    fetch_current_industry_membership_snapshot,
    fetch_sina_current_industry_membership_snapshot,
    fetch_eastmoney_individual_industry_membership_snapshot,
)


class FakeArchiveAkshare:
    def index_zh_a_hist(self, symbol, period, start_date, end_date):
        return pd.DataFrame([{
            "日期": "2026-11-10", "开盘": 3000, "最高": 3050, "最低": 2990,
            "收盘": 3030, "成交量": 100000, "成交额": 200000000,
        }])

    def stock_zh_a_spot_em(self):
        return pd.DataFrame([
            {"代码": "000001", "最新价": 10, "涨跌幅": 2.0, "成交额": 100.0},
            {"代码": "600000", "最新价": 8, "涨跌幅": -1.0, "成交额": 200.0},
            {"代码": "000002", "最新价": 9, "涨跌幅": 0.0, "成交额": 300.0},
        ])


    def stock_zt_pool_em(self, date):
        return pd.DataFrame([{"代码": "000001", "名称": "样例", "连板数": 2}])

    def stock_zt_pool_dtgc_em(self, date):
        return pd.DataFrame(columns=["代码", "名称"])

    def stock_zt_pool_zbgc_em(self, date):
        return pd.DataFrame([{"代码": "600000", "名称": "样例炸板", "炸板次数": 1}])

    def stock_zt_pool_previous_em(self, date):
        return pd.DataFrame([{"代码": "000002", "名称": "昨日涨停"}])

    def stock_zt_pool_strong_em(self, date):
        return pd.DataFrame([{"代码": "000003", "名称": "强势股"}])

    def stock_board_industry_name_em(self):
        return pd.DataFrame([
            {"板块名称": "电子", "板块代码": "BK0001"},
            {"板块名称": "计算机", "板块代码": "BK0002"},
        ])

    def stock_board_industry_cons_em(self, symbol):
        assert symbol in {"BK0001", "BK0002"}
        return pd.DataFrame({"代码": ["000001", "600000", "000002"]})


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
        observation_clock=lambda: pd.Timestamp("2026-11-11T16:30:00+08:00"),
    )
    assert set(counts) == {
        "index_000001", "index_399001", "index_399006",
        "index_000688", "index_000300", "industry_membership", "industry_membership_sources",
        "market_breadth", "limit_pools", "market_sentiment",
        "market_breadth_history_backfill", "market_sentiment_history_backfill",
    }
    assert all(counts[f"index_{code}"] == 1 for code in ("000001", "399001", "399006", "000688", "000300"))
    assert counts["industry_membership"] == 6  # selected taxonomy: three symbols across two industries
    assert counts["industry_membership_sources"] == 6
    assert counts["market_breadth"] == 1
    assert counts["market_sentiment"] == 1
    index_file = tmp_path / "index_daily" / "000001.jsonl"
    index_row = json.loads(index_file.read_text(encoding="utf-8").splitlines()[0])
    assert index_row["historical_pit_eligible"] is False
    assert index_row["availability_basis"] == "retrieval_time_only"
    assert index_row["available_time"] == "2026-11-11T08:30:00.000Z"

    industry_file = tmp_path / "industry_membership" / "2026-11-11.jsonl"
    industry_rows = [json.loads(line) for line in industry_file.read_text(encoding="utf-8").splitlines()]
    assert len(industry_rows) == 6
    assert {row["snapshot_scope"] for row in industry_rows} == {
        "current_membership_observation_only"
    }
    assert all(row["historical_pit_eligible"] is False for row in industry_rows)
    assert {row["availability_basis"] for row in industry_rows} == {"retrieval_time_only"}
    assert len({row["snapshot_id"] for row in industry_rows}) == 1
    assert all(row["snapshot_id"] for row in industry_rows)
    assert all("valid_from" not in row and "valid_to" not in row for row in industry_rows)

    sentiment_file = tmp_path / "market_sentiment" / "2026-11-11.jsonl"
    sentiment_row = json.loads(sentiment_file.read_text(encoding="utf-8").splitlines()[0])
    assert sentiment_row["highest_observed_limit_up_streak"] == 2
    assert sentiment_row["limit_up_count"] == 1
    assert sentiment_row["opened_rate"] is None
    assert sentiment_row["scoring_eligible"] is False

    breadth_file = tmp_path / "market_breadth" / "2026-11-11.jsonl"
    breadth_row = json.loads(breadth_file.read_text(encoding="utf-8").splitlines()[0])
    assert breadth_row["advances"] == 1
    assert breadth_row["declines"] == 1
    assert breadth_row["unchanged"] == 1
    assert breadth_row["breadth"] == pytest.approx(1 / 3)
    assert breadth_row["total_amount"] == 600.0
    assert breadth_row["total_amount_status"] == "complete_provider_snapshot_sum"
    assert breadth_row["snapshot_scope"] == "current_provider_snapshot_only"
    assert breadth_row["historical_pit_eligible"] is False



def test_market_observation_times_are_stamped_after_each_provider_response(tmp_path):
    start = pd.Timestamp("2026-11-12T08:30:00Z")
    observed = iter([start + pd.Timedelta(seconds=i) for i in range(11)])
    counts = collect(
        FakeArchiveAkshare(), tmp_path,
        captured_at="2026-11-12T08:30:00Z",
        observation_clock=lambda: next(observed),
        industry_clock=lambda: start + pd.Timedelta(seconds=99),
    )
    assert counts["index_000001"] == 1
    index_row = json.loads((tmp_path / "index_daily" / "000001.jsonl").read_text().splitlines()[0])
    breadth_row = json.loads((tmp_path / "market_breadth" / "2026-11-12.jsonl").read_text().splitlines()[0])
    pool_rows = [json.loads(line) for line in (tmp_path / "limit_pools" / "2026-11-12.jsonl").read_text().splitlines()]
    assert pd.Timestamp(index_row["available_time"]) == start
    assert pd.Timestamp(breadth_row["available_time"]) == start + pd.Timedelta(seconds=5)
    assert sorted(pd.Timestamp(row["capture_time"]) for row in pool_rows) == sorted(
        start + pd.Timedelta(seconds=i) for i in range(6, 11)
    )
    sentiment = json.loads((tmp_path / "market_sentiment" / "2026-11-12.jsonl").read_text().splitlines()[0])
    assert pd.Timestamp(sentiment["capture_time"]) == start + pd.Timedelta(seconds=10)


def test_collector_dry_run_validates_without_writing(tmp_path):
    counts = collect(
        FakeArchiveAkshare(),
        tmp_path,
        captured_at="2026-11-11T16:30:00+08:00",
        write=False,
    )
    assert counts["limit_pools"] == 5
    assert counts["market_sentiment"] == 1
    assert counts["market_breadth"] == 1
    assert list(tmp_path.rglob("*.jsonl")) == []


def test_collector_refuses_naive_capture_time(tmp_path):
    with pytest.raises(ValueError, match="timezone-aware"):
        collect(FakeArchiveAkshare(), tmp_path, captured_at="2026-11-11T08:30:00")



def test_forward_archive_retries_transient_provider_timeout(tmp_path):
    provider = FakeArchiveAkshare()
    original = provider.index_zh_a_hist
    calls = {"count": 0}

    def flaky_index(**kwargs):
        calls["count"] += 1
        if calls["count"] < 3:
            raise __import__("requests").exceptions.ReadTimeout("temporary TLS timeout")
        return original(**kwargs)

    provider.index_zh_a_hist = flaky_index
    counts = collect(
        provider, tmp_path, captured_at="2026-11-11T16:30:00+08:00",
        retry_sleep=lambda _: None,
    )
    assert calls["count"] == 7  # two failures plus success, then four remaining indexes
    assert counts["index_000001"] == 1


def test_forward_archive_does_not_retry_schema_or_data_errors():
    calls = {"count": 0}

    def invalid_provider():
        calls["count"] += 1
        raise ValueError("provider schema changed")

    with pytest.raises(ValueError, match="schema changed"):
        _call_with_network_retry(invalid_provider, sleep_fn=lambda _: None)
    assert calls["count"] == 1


def test_forward_archive_retries_only_up_to_configured_attempts():
    calls = {"count": 0}

    def unavailable_provider():
        calls["count"] += 1
        raise __import__("requests").exceptions.ReadTimeout("still unavailable")

    with pytest.raises(__import__("requests").exceptions.ReadTimeout):
        _call_with_network_retry(
            unavailable_provider, attempts=3, sleep_fn=lambda _: None
        )
    assert calls["count"] == 3



def test_same_day_repeated_industry_captures_are_archived_as_separate_batches(tmp_path):
    captured = "2026-11-11T16:30:00+08:00"
    first = collect(FakeArchiveAkshare(), tmp_path, captured_at=captured)
    second = collect(FakeArchiveAkshare(), tmp_path, captured_at=captured)
    assert first["industry_membership"] == 6
    assert second["industry_membership"] == 6
    path = tmp_path / "industry_membership" / "2026-11-11.jsonl"
    archived = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(archived) == 12
    assert len({row["snapshot_id"] for row in archived}) == 2

def test_industry_rows_record_each_response_completion_time(tmp_path):
    captured_times = iter([
        "2026-11-12T08:30:01Z",
        "2026-11-12T08:30:07Z",
    ])
    collect(
        FakeArchiveAkshare(),
        tmp_path,
        captured_at="2026-11-12T08:30:00Z",
        industry_clock=lambda: next(captured_times),
    )
    path = tmp_path / "industry_membership" / "2026-11-12.jsonl"
    archived = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    by_industry = {}
    for row in archived:
        by_industry.setdefault(row["industry_code"], set()).add(row["available_time"])
    assert len(by_industry) == 2
    assert {next(iter(values)) for values in by_industry.values()} == {
        "2026-11-12T08:30:01.000Z",
        "2026-11-12T08:30:07.000Z",
    }
    assert len({row["snapshot_id"] for row in archived}) == 1


def test_industry_availability_clock_is_captured_before_throttle_sleep():
    events = []
    times = iter([
        "2026-11-12T08:30:01Z",
        "2026-11-12T08:30:07Z",
    ])

    class OrderedProvider(FakeArchiveAkshare):
        def stock_board_industry_cons_em(self, symbol):
            events.append(("response", symbol))
            return super().stock_board_industry_cons_em(symbol)

    def clock():
        events.append(("clock", None))
        return next(times)

    def sleep_fn(seconds):
        events.append(("sleep", seconds))

    fetch_current_industry_membership_snapshot(
        OrderedProvider(), clock=clock, sleep_fn=sleep_fn
    )

    assert [kind for kind, _ in events] == [
        "response", "clock", "sleep",
        "response", "clock", "sleep",
    ]
    assert [value for kind, value in events if kind == "sleep"] == [1.0, 1.0]


def test_collector_uses_tencent_index_fallback_after_push2_rate_limit(tmp_path):
    class RateLimitedIndexProvider(FakeArchiveAkshare):
        def __init__(self):
            self.tencent_calls = []

        def index_zh_a_hist(self, symbol, period, start_date, end_date):
            raise type("Push2RateLimitError", (Exception,), {})("cooldown")

        def stock_zh_index_daily_tx(self, symbol, start_date, end_date):
            self.tencent_calls.append(symbol)
            return pd.DataFrame([{
                "date": "2026-11-11", "open": 3000, "high": 3050,
                "low": 2990, "close": 3030, "volume": 100000, "amount": 250000000,
            }])

    provider = RateLimitedIndexProvider()
    counts = collect(
        provider, tmp_path, captured_at="2026-11-12T08:30:00Z"
    )

    assert provider.tencent_calls == [
        "sh000001", "sz399001", "sz399006", "sh000688", "sh000300"
    ]
    assert counts["index_000001"] == 1
    archived = json.loads(
        (tmp_path / "index_daily" / "000001.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    assert archived["volume"] == 100000
    assert archived["amount"] == 250000000



def test_partial_archive_uses_tencent_breadth_when_eastmoney_is_rate_limited(tmp_path):
    individual_calls = []

    class PartiallyRateLimitedProvider(FakeArchiveAkshare):
        def stock_individual_info_em(self, symbol):
            individual_calls.append(symbol)
            raise AssertionError("per-stock Eastmoney calls must be skipped during cooldown")
        def stock_board_industry_name_em(self):
            raise type("Push2RateLimitError", (Exception,), {})("Eastmoney cooldown")

        def stock_zh_a_spot_em(self):
            raise type("Push2RateLimitError", (Exception,), {})("Eastmoney cooldown")

        def stock_zh_a_spot_tx(self):
            return pd.DataFrame([
                {"code": "sz000001", "zxj": 10, "zdf": 2.0, "turnover": 100.0},
                {"code": "sh600000", "zxj": 8, "zdf": -1.0, "turnover": 200.0},
            ])

    counts = collect(
        PartiallyRateLimitedProvider(),
        tmp_path,
        captured_at="2026-11-12T08:30:00Z",
        allow_partial=True,
    )
    assert counts["index_000001"] == 1
    assert counts["market_breadth"] == 1
    assert counts["industry_membership"] == 0
    breadth = json.loads(
        (tmp_path / "market_breadth" / "2026-11-12.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    assert breadth["provider_row_count"] == 2
    assert breadth["advances"] == 1
    assert breadth["declines"] == 1
    assert breadth["total_amount"] == 3000000.0
    assert breadth["source_id"] == "tencent.stock_zh_a_spot_tx"
    report = json.loads(
        (tmp_path / "collection_report" / "2026-11-12.json").read_text(encoding="utf-8")
    )
    assert report["partial"] is True
    assert report["source_status"]["industry_membership"].startswith("unavailable:")
    assert "Push2RateLimitError: Eastmoney cooldown" in report["source_status"]["industry_membership"]
    assert "skipped_due_to_eastmoney_rate_limit" in report["source_status"]["industry_membership"]
    assert individual_calls == []
    assert report["source_status"]["market_breadth"] == "ok: tencent.stock_zh_a_spot_tx"



def test_collector_uses_official_shenwan_when_existing_taxonomies_are_partial(tmp_path):
    class PartialTaxonomyProvider(FakeArchiveAkshare):
        def stock_board_industry_name_em(self):
            raise type("Push2RateLimitError", (Exception,), {})("Eastmoney cooldown")

        def stock_sector_spot(self, indicator):
            return pd.DataFrame([
                {"label": f"industry_{index:02d}", "板块": f"行业{index}"}
                for index in range(10)
            ])

        def stock_sector_detail(self, sector):
            return pd.DataFrame([
                {"code": "000001", "name": "平安银行"},
                {"code": "600000", "name": "浦发银行"},
            ])

    captured = pd.Timestamp("2026-11-13T14:00:00Z")
    sw_snapshot = pd.DataFrame([
        {
            "symbol": symbol,
            "industry_code": code,
            "industry_name": name,
            "snapshot_id": "sw-snapshot",
            "observed_at": captured,
            "available_time": captured,
            "source_id": "swsresearch.sw2021_stock_classification",
            "snapshot_scope": "current_membership_observation_only",
            "source_revision": f"revision-{symbol}",
        }
        for symbol, code, name in [
            ("000001", "480101", "银行"),
            ("600000", "480101", "银行"),
            ("000002", "430101", "房地产"),
        ]
    ])

    counts = collect(
        PartialTaxonomyProvider(),
        tmp_path,
        captured_at="2026-11-13T16:30:00+08:00",
        allow_partial=True,
        sw_industry_fetcher=lambda clock=None: sw_snapshot,
    )
    report = json.loads(
        (tmp_path / "collection_report" / "2026-11-13.json").read_text(encoding="utf-8")
    )
    assert counts["industry_membership"] == 3
    assert report["industry_audit"]["selected_source"] == "swsresearch.sw2021_stock_classification"
    assert report["industry_audit"]["shenwan_coverage"] == 1.0
    assert report["industry_audit"]["selected_snapshot_coverage"] == 1.0
    assert report["source_status"]["industry_membership"].startswith("ok: swsresearch.sw2021_stock_classification")

def test_collector_uses_sina_industry_fallback_when_eastmoney_is_rate_limited(tmp_path):
    class SinaFallbackProvider(FakeArchiveAkshare):
        def stock_board_industry_name_em(self):
            raise type("Push2RateLimitError", (Exception,), {})("Eastmoney cooldown")

        def stock_sector_spot(self, indicator):
            assert indicator == "新浪行业"
            return pd.DataFrame([
                {"label": f"industry_{index:02d}", "板块": f"行业{index}"}
                for index in range(10)
            ])

        def stock_sector_detail(self, sector):
            index = int(sector.rsplit("_", 1)[1])
            return pd.DataFrame([
                {"code": "000001", "name": "平安银行"},
                {"code": f"6000{index:02d}", "name": f"样本股票{index}"},
                {"code": "000002", "name": "样例三"},
            ])

    counts = collect(
        SinaFallbackProvider(),
        tmp_path,
        captured_at="2026-11-12T08:30:00Z",
        allow_partial=True,
    )
    assert counts["industry_membership"] == 30
    report = json.loads(
        (tmp_path / "collection_report" / "2026-11-12.json").read_text(encoding="utf-8")
    )
    assert report["source_status"]["industry_membership"].startswith(
        "ok: akshare.stock_sector_detail_sina"
    )
    path = tmp_path / "industry_membership" / "2026-11-12.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 30
    assert {row["source_id"] for row in rows} == {"akshare.stock_sector_detail_sina"}
    assert {row["expected_industry_count"] for row in rows} == {10}
    assert all(row["historical_pit_eligible"] is False for row in rows)
    assert all("valid_from" not in row and "valid_to" not in row for row in rows)


def test_sina_industry_rows_record_each_sector_response_completion_time():
    class SinaProvider:
        def stock_sector_spot(self, indicator):
            assert indicator == "新浪行业"
            return pd.DataFrame([
                {"label": f"industry_{index:02d}", "板块": f"行业{index}"}
                for index in range(10)
            ])

        def stock_sector_detail(self, sector):
            index = int(sector.rsplit("_", 1)[1])
            return pd.DataFrame([
                {"code": "000001", "name": "平安银行"},
                {"code": f"6000{index:02d}", "name": f"样本股票{index}"},
            ])

    start = pd.Timestamp("2026-11-12T08:30:00Z")
    times = iter(start + pd.Timedelta(seconds=i) for i in range(10))
    snapshot = fetch_sina_current_industry_membership_snapshot(
        SinaProvider(), clock=lambda: next(times)
    )

    by_industry = snapshot.groupby("industry_code")["available_time"].first()
    assert len(by_industry) == 10
    assert sorted(by_industry.tolist()) == sorted(
        start + pd.Timedelta(seconds=i) for i in range(10)
    )
    assert snapshot.groupby("industry_code")["available_time"].nunique().eq(1).all()
    assert snapshot["snapshot_id"].nunique() == 1
    assert snapshot["historical_pit_eligible"].eq(False).all()



def test_collection_report_records_each_index_provider_and_row_count(tmp_path):
    collect(
        FakeArchiveAkshare(),
        tmp_path,
        captured_at="2026-11-12T16:30:00+08:00",
        allow_partial=True,
    )
    report = json.loads(
        (tmp_path / "collection_report" / "2026-11-12.json").read_text(encoding="utf-8")
    )
    for code in ("000001", "399001", "399006", "000688", "000300"):
        assert report["source_status"][f"index_{code}"] == (
            "ok: akshare.index_zh_a_hist; rows=1"
        )
    assert report["partial"] is False



def test_sina_industry_snapshot_collapses_exact_duplicate_memberships():
    class FakeSina:
        def stock_sector_spot(self, indicator):
            return pd.DataFrame([
                {"label": f"IND-{i}", "板块": f"行业{i}"} for i in range(10)
            ])

        def stock_sector_detail(self, sector):
            # Repeated rows within an industry are provider duplication;
            # the same symbol in different industries remains valid.
            return pd.DataFrame([
                {"code": "000001", "name": "平安银行"},
                {"code": "000001", "name": "平安银行"},
                {"code": "600000", "name": "浦发银行"},
            ])

    result = fetch_sina_current_industry_membership_snapshot(
        FakeSina(), clock=lambda: pd.Timestamp("2026-11-13T08:00:00Z")
    )
    assert len(result) == 20
    assert not result.duplicated(["symbol", "industry_code"]).any()
    assert result.groupby("symbol")["industry_code"].nunique().loc["000001"] == 10
    assert result["historical_pit_eligible"].eq(False).all()


def test_sina_industry_snapshot_rejects_conflicting_duplicate_names():
    class FakeSina:
        def stock_sector_spot(self, indicator):
            return pd.DataFrame([
                {"label": f"IND-{i}", "板块": f"行业{i}"} for i in range(10)
            ])

        def stock_sector_detail(self, sector):
            return pd.DataFrame([
                {"code": "000001", "name": "名称A"},
                {"code": "000001", "name": "名称B"},
            ])

    with pytest.raises(ValueError, match="conflicting names"):
        fetch_sina_current_industry_membership_snapshot(
            FakeSina(), clock=lambda: pd.Timestamp("2026-11-13T08:00:00Z")
        )


def test_collector_prefers_sina_when_eastmoney_industry_coverage_is_incomplete(tmp_path):
    class IncompleteEastmoneyProvider(FakeArchiveAkshare):
        def stock_board_industry_cons_em(self, symbol):
            return pd.DataFrame({"代码": ["000001"]})

        def stock_sector_spot(self, indicator):
            return pd.DataFrame([
                {"label": f"industry_{index:02d}", "板块": f"行业{index}"}
                for index in range(10)
            ])

        def stock_sector_detail(self, sector):
            return pd.DataFrame([
                {"code": "000001", "name": "样例一"},
                {"code": "600000", "name": "样例二"},
                {"code": "000002", "name": "样例三"},
            ])

    provider = IncompleteEastmoneyProvider()
    counts = collect(
        provider, tmp_path,
        captured_at="2026-11-13T16:30:00+08:00",
        allow_partial=True,
    )
    assert counts["industry_membership"] == 30
    report = json.loads(
        (tmp_path / "collection_report" / "2026-11-13.json").read_text(encoding="utf-8")
    )
    assert report["source_status"]["industry_membership"] == (
        "ok: akshare.stock_sector_detail_sina; symbol_coverage=100.0%"
    )
    rows = [
        json.loads(line)
        for line in (tmp_path / "industry_membership" / "2026-11-13.jsonl").read_text().splitlines()
    ]
    assert {row["source_id"] for row in rows} == {"akshare.stock_sector_detail_sina"}
    assert {row["symbol"] for row in rows} == {"000001", "600000", "000002"}



def test_collector_archives_all_industry_sources_and_reports_union_coverage(tmp_path):
    class SplitCoverageProvider(FakeArchiveAkshare):
        def stock_board_industry_cons_em(self, symbol):
            return pd.DataFrame({"代码": ["000001"]})

        def stock_sector_spot(self, indicator):
            return pd.DataFrame([
                {"label": f"industry_{index:02d}", "板块": f"行业{index}"}
                for index in range(10)
            ])

        def stock_sector_detail(self, sector):
            return pd.DataFrame([
                {"code": "000001", "name": "样例一"},
                {"code": "600000", "name": "样例二"},
            ])

    def baidu_fetcher(symbols):
        assert symbols == ["000001", "000002", "600000"]
        return pd.DataFrame([{
            "symbol": "000002",
            "industry_code": "BD-TECH",
            "industry_name": "百度行业样例",
            "industry_level1": "百度分类",
            "industry_level2": "百度行业样例",
            "snapshot_id": "baidu-snapshot",
            "expected_industry_count": 1,
            "observed_at": "2026-11-13T08:00:00Z",
            "available_time": "2026-11-13T08:00:00Z",
            "availability_basis": "retrieval_time_only",
            "historical_pit_eligible": False,
            "source_id": "baidu.gushitong.relatedblock",
            "snapshot_scope": "current_membership_observation_only",
            "source_revision": "baidu-revision",
        }])

    counts = collect(
        SplitCoverageProvider(), tmp_path,
        captured_at="2026-11-13T16:30:00+08:00",
        allow_partial=True,
        baidu_industry_fetcher=baidu_fetcher,
    )
    assert counts["industry_membership_sources"] > counts["industry_membership"]
    report = json.loads(
        (tmp_path / "collection_report" / "2026-11-13.json").read_text(encoding="utf-8")
    )
    audit = report["industry_audit"]
    assert audit["quote_universe_symbol_count"] == 3
    assert audit["multi_source_covered_symbol_count"] == 3
    assert audit["multi_source_coverage"] == 1.0
    assert audit["uncovered_symbols"] == []
    raw_rows = [
        json.loads(line)
        for line in (tmp_path / "industry_membership_sources" / "2026-11-13.jsonl").read_text().splitlines()
    ]
    assert {"akshare.stock_board_industry_cons_em", "akshare.stock_sector_detail_sina",
            "baidu.gushitong.relatedblock"}.issubset({row["source_id"] for row in raw_rows})
    assert {row["symbol"] for row in raw_rows} == {"000001", "000002", "600000"}
    # The selected model-consumption snapshot remains one taxonomy only.
    selected_rows = [
        json.loads(line)
        for line in (tmp_path / "industry_membership" / "2026-11-13.jsonl").read_text().splitlines()
    ]
    assert len({row["source_id"] for row in selected_rows}) == 1


def test_collector_backfills_missing_historical_market_context(tmp_path):
    def historical_fetcher(trade_date, *, expected_market_count):
        assert trade_date == "2026-11-10"
        assert expected_market_count == 3
        breadth = {
            "trade_date": trade_date,
            "snapshot_time": "2026-11-10T07:40:00+00:00",
            "available_time": "2026-11-12T08:30:00+00:00",
            "breadth": 0.6,
            "breadth_coverage": 1.0,
            "total_amount": 1000000000,
            "total_amount_status": "complete_provider_market_aggregate",
            "source_id": "kaipanhong.market_emotion_kph",
            "source_revision": "breadth-revision",
            "historical_pit_eligible": False,
        }
        sentiment = {
            "trade_date": trade_date,
            "capture_time": "2026-11-10T07:40:00+00:00",
            "available_time": "2026-11-12T08:30:00+00:00",
            "limit_up_count": 20,
            "limit_down_count": 3,
            "source_id": "kaipanhong.market_emotion_kph",
            "source_revision": "sentiment-revision",
            "historical_pit_eligible": False,
        }
        return breadth, sentiment

    counts = collect(
        FakeArchiveAkshare(),
        tmp_path,
        captured_at="2026-11-12T08:30:00Z",
        allow_partial=True,
        historical_market_fetcher=historical_fetcher,
    )
    assert counts["market_breadth_history_backfill"] == 1
    assert counts["market_sentiment_history_backfill"] == 1
    breadth = json.loads(
        (tmp_path / "market_breadth" / "2026-11-10.jsonl").read_text().splitlines()[0]
    )
    sentiment = json.loads(
        (tmp_path / "market_sentiment" / "2026-11-10.jsonl").read_text().splitlines()[0]
    )
    assert breadth["source_id"] == "kaipanhong.market_emotion_kph"
    assert breadth["available_time"] == "2026-11-12T08:30:00+00:00"
    assert sentiment["limit_up_count"] == 20
    assert sentiment["historical_pit_eligible"] is False



class FakeBaostockResult:
    fields = ["updateDate", "code", "code_name", "industry", "industryClassification"]

    def __init__(self, rows):
        self.rows = list(rows)
        self.index = 0
        self.error_code = "0"
        self.error_msg = ""

    def next(self):
        self.index += 1
        return self.index <= len(self.rows)

    def get_row_data(self):
        return self.rows[self.index - 1]


class FakeBaostock:
    def __init__(self, rows):
        self.rows = rows
        self.logged_out = False

    def login(self):
        return SimpleNamespace(error_code="0", error_msg="")

    def query_stock_industry(self):
        return FakeBaostockResult(self.rows)

    def logout(self):
        self.logged_out = True


def test_baostock_industry_snapshot_preserves_update_date_and_retrieval_time():
    provider = FakeBaostock([
        ["2026-11-09", "sh.600519", "贵州茅台", "食品饮料", "申万一级行业"],
        ["2026-11-09", "sz.000858", "五粮液", "食品饮料", "申万一级行业"],
        ["2026-11-09", "sh.600036", "招商银行", "银行", "申万一级行业"],
    ])
    captured = pd.Timestamp("2026-11-13T12:00:00Z")
    result = fetch_baostock_industry_membership_snapshot(
        provider, clock=lambda: captured
    )

    assert provider.logged_out is True
    assert set(result["symbol"]) == {"600519", "000858", "600036"}
    assert set(result["updateDate"]) == {"2026-11-09"}
    assert result["available_time"].eq(captured).all()
    assert result["industryClassification"].eq("申万一级行业").all()
    assert result["historical_pit_eligible"].eq(False).all()
    assert result["source_id"].eq("baostock.query_stock_industry").all()


def test_baostock_industry_snapshot_rejects_mixed_classification_standards():
    provider = FakeBaostock([
        ["2026-11-09", "sh.600519", "贵州茅台", "食品饮料", "申万一级行业"],
        ["2026-11-09", "sz.000858", "五粮液", "C15酒、饮料和精制茶制造业", "证监会行业分类"],
    ])
    with pytest.raises(ValueError, match="mixes industry classification"):
        fetch_baostock_industry_membership_snapshot(
            provider, clock=lambda: pd.Timestamp("2026-11-13T12:00:00Z")
        )


def test_collector_uses_baostock_when_eastmoney_and_sina_industry_coverage_are_partial(tmp_path):
    class PartialTaxonomyAkshare(FakeArchiveAkshare):
        def stock_board_industry_cons_em(self, symbol):
            return pd.DataFrame({"代码": ["000001"]})

        def stock_sector_spot(self, indicator):
            return pd.DataFrame([
                {"label": f"industry_{index:02d}", "板块": f"行业{index}"}
                for index in range(10)
            ])

        def stock_sector_detail(self, sector):
            return pd.DataFrame([
                {"code": "000001", "name": "平安银行"},
                {"code": "600000", "name": "浦发银行"},
            ])

    provider = FakeBaostock([
        ["2026-11-09", "sz.000001", "平安银行", "银行", "申万一级行业"],
        ["2026-11-09", "sh.600000", "浦发银行", "银行", "申万一级行业"],
        ["2026-11-09", "sz.000002", "万科A", "房地产", "申万一级行业"],
    ])
    collect(
        PartialTaxonomyAkshare(), tmp_path,
        captured_at="2026-11-13T16:30:00+08:00",
        allow_partial=True, baostock_module=provider,
    )
    report = json.loads(
        (tmp_path / "collection_report" / "2026-11-13.json").read_text(encoding="utf-8")
    )
    assert report["source_status"]["industry_membership"] == (
        "ok: baostock.query_stock_industry; symbol_coverage=100.0%; "
        "Eastmoney_board=33.3%; Eastmoney_individual=not_available; individual_failures=0; Sina=66.7%; Baostock=100.0%; Shenwan=not_available"
    )
    rows = [
        json.loads(line)
        for line in (tmp_path / "industry_membership" / "2026-11-13.jsonl").read_text().splitlines()
    ]
    assert {row["source_id"] for row in rows} == {"baostock.query_stock_industry"}
    assert {row["industryClassification"] for row in rows} == {"申万一级行业"}
    assert all(row["historical_pit_eligible"] is False for row in rows)


def test_eastmoney_individual_industry_adapter_stamps_each_response_and_keeps_failures():
    class IndividualProvider:
        def stock_individual_info_em(self, symbol):
            if symbol == "000003":
                raise RuntimeError("temporary unavailable")
            industry = {"000001": "银行", "000002": "电子"}[symbol]
            return pd.DataFrame([
                {"item": "股票代码", "value": symbol},
                {"item": "行业", "value": industry},
            ])

    start = pd.Timestamp("2026-11-13T12:00:00Z")
    times = iter(start + pd.Timedelta(seconds=i) for i in range(2))
    result = fetch_eastmoney_individual_industry_membership_snapshot(
        IndividualProvider(), ["000001", "000002", "000003"],
        clock=lambda: next(times), pause_seconds=0, sleep_fn=lambda _: None,
        max_attempts=1,
    )
    assert set(result["symbol"]) == {"000001", "000002"}
    assert result["available_time"].nunique() == 2
    assert result["historical_pit_eligible"].eq(False).all()
    assert result.attrs["requested_symbol_count"] == 3
    assert result.attrs["success_symbol_count"] == 2
    assert set(result.attrs["failed_symbols"]) == {"000003"}


def test_collector_fills_eastmoney_industry_gaps_from_individual_profiles(tmp_path):
    class GapProvider(FakeArchiveAkshare):
        def stock_board_industry_cons_em(self, symbol):
            if symbol == "BK0001":
                return pd.DataFrame({"代码": ["000001"]})
            return pd.DataFrame({"代码": []})

        def stock_individual_info_em(self, symbol):
            categories = {"600000": "银行", "000002": "计算机"}
            return pd.DataFrame([
                {"item": "股票代码", "value": symbol},
                {"item": "行业", "value": categories[symbol]},
            ])

        def stock_sector_spot(self, indicator):
            raise AssertionError("Sina taxonomy must not be mixed into completed Eastmoney coverage")

    counts = collect(
        GapProvider(), tmp_path,
        captured_at="2026-11-13T16:30:00+08:00",
        allow_partial=True,
        retry_sleep=lambda _: None,
    )
    assert counts["industry_membership"] == 3
    report = json.loads(
        (tmp_path / "collection_report" / "2026-11-13.json").read_text(encoding="utf-8")
    )
    assert report["source_status"]["industry_membership"].startswith(
        "ok: eastmoney_board_plus_individual_info; symbol_coverage=100.0%"
    )
    rows = [
        json.loads(line)
        for line in (tmp_path / "industry_membership" / "2026-11-13.jsonl").read_text().splitlines()
    ]
    assert {row["symbol"] for row in rows} == {"000001", "600000", "000002"}
    assert rows[1]["source_id"] in {
        "akshare.stock_board_industry_cons_em", "akshare.stock_individual_info_em"
    }
    assert all(row["historical_pit_eligible"] is False for row in rows)

