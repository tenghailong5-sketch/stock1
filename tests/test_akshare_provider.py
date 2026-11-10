import pandas as pd
import pytest

from src.providers.akshare_pit import (
    fetch_current_industry_membership_snapshot,
    normalize_daily_history_snapshot,
    normalize_index_history_snapshot,
)


def test_daily_history_uses_retrieval_time_not_trade_date_as_availability():
    raw = pd.DataFrame([{
        "日期": "2024-01-02", "开盘": 10, "最高": 11, "最低": 9,
        "收盘": 10.5, "成交量": 100, "成交额": 1000, "换手率": 1.2,
    }])
    out = normalize_daily_history_snapshot(
        raw, symbol="1", fetched_at="2026-11-11T08:00:00Z"
    )
    assert out.iloc[0].symbol == "000001"
    assert out.iloc[0].available_time == pd.Timestamp("2026-11-11T08:00:00Z")
    assert out.iloc[0].trade_date == pd.Timestamp("2024-01-02")
    assert out.iloc[0].historical_pit_eligible == False
    assert out.iloc[0].availability_basis == "retrieval_time_only"
    assert len(out.iloc[0].source_revision) == 64


def test_daily_history_rejects_schema_drift_and_duplicates():
    raw = pd.DataFrame([{
        "日期": "2024-01-02", "开盘": 10, "最高": 11, "最低": 9,
        "收盘": 10.5, "成交量": 100, "成交额": 1000,
    }])
    with pytest.raises(ValueError, match="schema missing"):
        normalize_daily_history_snapshot(raw, symbol="000001")
    raw["换手率"] = 1.0
    raw = pd.concat([raw, raw], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        normalize_daily_history_snapshot(raw, symbol="000001")


class FakeAkshare:
    def stock_board_industry_name_em(self):
        return pd.DataFrame([
            {"板块名称": "电子", "板块代码": "BK0001"},
            {"板块名称": "计算机", "板块代码": "BK0002"},
        ])

    def stock_board_industry_cons_em(self, symbol):
        return pd.DataFrame({"代码": ["000001", "600000"]})


def test_industry_membership_is_explicitly_snapshot_only():
    ticks = iter([
        "2026-11-11T08:00:00Z",
        "2026-11-11T08:00:01Z",
    ])
    out = fetch_current_industry_membership_snapshot(
        FakeAkshare(), clock=lambda: next(ticks)
    )
    assert len(out) == 4
    assert set(out.snapshot_scope) == {"current_membership_observation_only"}
    assert "valid_from" not in out.columns
    assert "valid_to" not in out.columns
    assert out.symbol.str.len().eq(6).all()


def test_industry_membership_fails_closed_on_provider_schema_change():
    class Broken(FakeAkshare):
        def stock_board_industry_cons_em(self, symbol):
            return pd.DataFrame({"证券代码": ["000001"]})

    with pytest.raises(ValueError, match="schema changed"):
        fetch_current_industry_membership_snapshot(Broken())

 

def test_daily_history_revision_hash_ignores_retrieval_timestamp():
    raw = pd.DataFrame([{
        "日期": "2024-01-02", "开盘": 10, "最高": 11, "最低": 9,
        "收盘": 10.5, "成交量": 100, "成交额": 1000, "换手率": 1.2,
    }])
    first = normalize_daily_history_snapshot(raw, symbol="000001", fetched_at="2026-11-11T08:00:00Z")
    second = normalize_daily_history_snapshot(raw, symbol="000001", fetched_at="2026-11-12T08:00:00Z")
    assert first.iloc[0].source_revision == second.iloc[0].source_revision


def test_index_history_is_normalized_and_never_claims_historical_pit():
    raw = pd.DataFrame([{
        "日期": "2026-11-10", "开盘": 3000, "最高": 3050, "最低": 2990,
        "收盘": 3030, "成交量": 100000, "成交额": 200000000,
    }])
    out = normalize_index_history_snapshot(
        raw, symbol="000001", fetched_at="2026-11-11T08:00:00Z"
    )
    assert out.iloc[0].index_symbol == "000001"
    assert out.iloc[0].source_id == "akshare.index_zh_a_hist"
    assert out.iloc[0].historical_pit_eligible == False
    assert out.iloc[0].available_time == pd.Timestamp("2026-11-11T08:00:00Z")


def test_index_history_rejects_invalid_schema_and_ohlc():
    raw = pd.DataFrame([{
        "日期": "2026-11-10", "开盘": 3000, "最高": 2990, "最低": 2995,
        "收盘": 3030, "成交量": 100000, "成交额": 200000000,
    }])
    with pytest.raises(ValueError, match="invalid OHLC"):
        normalize_index_history_snapshot(raw, symbol="000001")


def test_daily_history_revision_is_per_row_not_per_rolling_window():
    raw = pd.DataFrame([
        {"日期": "2024-01-02", "开盘": 10, "最高": 11, "最低": 9,
         "收盘": 10.5, "成交量": 100, "成交额": 1000, "换手率": 1.2},
        {"日期": "2024-01-03", "开盘": 10.5, "最高": 12, "最低": 10,
         "收盘": 11.5, "成交量": 120, "成交额": 1300, "换手率": 1.4},
    ])
    first = normalize_daily_history_snapshot(raw, symbol="000001")
    changed = raw.copy()
    changed.loc[1, "收盘"] = 11.6
    second = normalize_daily_history_snapshot(changed, symbol="000001")
    first_hashes = dict(zip(first.trade_date.dt.strftime("%Y-%m-%d"), first.source_revision))
    second_hashes = dict(zip(second.trade_date.dt.strftime("%Y-%m-%d"), second.source_revision))
    assert first_hashes["2024-01-02"] == second_hashes["2024-01-02"]
    assert first_hashes["2024-01-03"] != second_hashes["2024-01-03"]


def test_market_breadth_snapshot_counts_valid_quotes_without_limit_inference():
    raw = pd.DataFrame([
        {"代码": "000001", "最新价": 10, "涨跌幅": 2.5},
        {"代码": "600000", "最新价": 8, "涨跌幅": -1.2},
        {"代码": "000002", "最新价": 9, "涨跌幅": 0},
        {"代码": "300001", "最新价": None, "涨跌幅": None},
    ])
    result = normalize_market_breadth_snapshot(
        raw, fetched_at="2026-11-11T16:30:00+08:00"
    ).iloc[0]
    assert result["provider_row_count"] == 4
    assert result["valid_quote_count"] == 3
    assert result["missing_quote_count"] == 1
    assert result["advances"] == 1
    assert result["declines"] == 1
    assert result["unchanged"] == 1
    assert result["breadth_coverage"] == 0.75
    assert result["historical_pit_eligible"] is False
    assert result["snapshot_scope"] == "current_provider_snapshot_only"


def test_market_breadth_snapshot_rejects_duplicate_symbols():
    raw = pd.DataFrame([
        {"代码": "000001", "最新价": 10, "涨跌幅": 1},
        {"代码": "000001", "最新价": 11, "涨跌幅": 2},
    ])
    with pytest.raises(ValueError, match="duplicate symbols"):
        normalize_market_breadth_snapshot(raw)
