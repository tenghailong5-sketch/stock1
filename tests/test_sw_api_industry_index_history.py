import pandas as pd
import pytest

from src.providers.sw_api_industry import (
    fetch_sw_api_first_level_index_list,
    fetch_sw_api_industry_index_history,
    INDEX_URL,
    TREND_URL,
)


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class FakeSession:
    def __init__(self, payloads):
        self.payloads = payloads

    def get(self, url, **kwargs):
        return FakeResponse(self.payloads[url])


def test_fetch_sw_first_level_index_list_requires_a_real_cross_section():
    rows = [
        {"swindexcode": f"80{i:04d}", "swindexname": f"industry-{i}"}
        for i in range(1, 22)
    ]
    session = FakeSession({
        INDEX_URL: {"data": {"results": rows, "count": len(rows)}}
    })
    result = fetch_sw_api_first_level_index_list(session=session)
    assert len(result) == 21
    assert result.industry_code.is_unique


def test_fetch_sw_index_history_normalizes_official_payload_and_provenance():
    rows = [{
        "swindexcode": "801010",
        "bargaindate": day.strftime("%Y-%m-%d"),
        "openindex": close - 1,
        "maxindex": close + 1,
        "minindex": close - 2,
        "closeindex": close,
        "bargainamount": 100,
        "bargainsum": 1000 + i * 10,
    } for i, (day, close) in enumerate(zip(
        pd.date_range("2026-11-02", periods=7), [100, 101, 102, 103, 104, 105, 110]
    ))]
    session = FakeSession({TREND_URL: {"data": rows}})
    result = fetch_sw_api_industry_index_history(
        "801010", start_date="20261102", end_date="20261108",
        session=session, clock=lambda: pd.Timestamp("2026-11-09T00:00:00Z"),
    )
    assert len(result) == 7
    assert result.industry_code.eq("801010").all()
    assert result.iloc[-1].return_3 == pytest.approx(110 / 104 - 1)
    assert result.iloc[-1].return_5 == pytest.approx(110 / 101 - 1)
    assert result.iloc[-1].amount == 1060
    assert result.historical_pit_eligible.eq(False).all()
    assert result.availability_basis.eq("retrieval_time_only").all()
    assert result.source_id.eq("swsresearch.index_publish.trend").all()


def test_fetch_sw_index_history_rejects_invalid_code_and_empty_data():
    with pytest.raises(ValueError, match="six-digit"):
        fetch_sw_api_industry_index_history(
            "80101", start_date="20261102", end_date="20261108",
            session=FakeSession({TREND_URL: {"data": []}}),
        )
    with pytest.raises(ValueError, match="empty or invalid"):
        fetch_sw_api_industry_index_history(
            "801010", start_date="20261102", end_date="20261108",
            session=FakeSession({TREND_URL: {"data": []}}),
        )
