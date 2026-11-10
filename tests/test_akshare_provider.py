import pandas as pd
import pytest

from src.providers.akshare_pit import (
    fetch_current_industry_membership_snapshot,
    normalize_daily_history_snapshot,
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
