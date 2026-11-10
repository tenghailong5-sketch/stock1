import pandas as pd
import pytest

from src.providers.akshare_limit_pools import (
    POOL_APIS,
    fetch_limit_pool_snapshots,
    normalize_limit_pool_snapshot,
)


def test_normalize_limit_pool_keeps_provider_fields_and_marks_forward_only():
    raw = pd.DataFrame([
        {"代码": "1", "名称": "测试", "连板数": 2, "首次封板时间": "09:31:00"}
    ])
    row = normalize_limit_pool_snapshot(
        raw, event_type="limit_up", trade_date="20261111",
        captured_at="2026-11-11T16:30:00+08:00",
    ).iloc[0]
    assert row["trade_date"] == "2026-11-11"
    assert row["records"][0]["代码"] == "000001"
    assert row["records"][0]["连板数"] == 2
    assert row["provider_row_count"] == 1
    assert row["historical_pit_eligible"] is False
    assert row["snapshot_scope"] == "date_scoped_provider_pool_snapshot"


def test_empty_pool_with_expected_schema_is_a_valid_zero_count_snapshot():
    row = normalize_limit_pool_snapshot(
        pd.DataFrame(columns=["代码", "名称"]),
        event_type="limit_down", trade_date="20261111",
    ).iloc[0]
    assert row["provider_row_count"] == 0
    assert row["records"] == []


@pytest.mark.parametrize("frame", [
    None,
    pd.DataFrame([{"名称": "missing code"}]),
])
def test_invalid_pool_response_fails_closed(frame):
    with pytest.raises(ValueError):
        normalize_limit_pool_snapshot(
            frame, event_type="limit_up", trade_date="20261111"
        )


def test_duplicate_symbols_are_rejected():
    raw = pd.DataFrame([{"代码": "000001"}, {"代码": "1"}])
    with pytest.raises(ValueError, match="duplicate"):
        normalize_limit_pool_snapshot(
            raw, event_type="limit_up", trade_date="20261111"
        )


def test_invalid_trade_date_is_rejected():
    with pytest.raises(ValueError, match="YYYYMMDD"):
        normalize_limit_pool_snapshot(
            pd.DataFrame(columns=["代码"]),
            event_type="limit_up", trade_date="2026-11-11",
        )


def test_fetch_calls_all_four_date_scoped_apis():
    class Fake:
        def __init__(self):
            self.calls = []

        def __getattr__(self, name):
            if name not in POOL_APIS.values():
                raise AttributeError(name)

            def call(*, date):
                self.calls.append((name, date))
                return pd.DataFrame(columns=["代码", "名称"])
            return call

    fake = Fake()
    result = fetch_limit_pool_snapshots(
        fake, trade_date="20261111",
        captured_at="2026-11-11T16:30:00+08:00",
    )
    assert len(result) == 4
    assert {event for event, _ in fake.calls} == set(POOL_APIS.values())
    assert {date for _, date in fake.calls} == {"20261111"}
