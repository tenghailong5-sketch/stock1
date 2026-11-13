import pytest

from src.providers.kaipanhong_market import fetch_historical_market_context


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


def test_historical_market_context_keeps_observation_and_retrieval_times_separate():
    calls = []

    def fake_post(url, *, headers, data, timeout):
        calls.append((url, data, timeout))
        return FakeResponse({
            "errcode": "0",
            "info": {
                "SZJS": "3000",
                "XDJS": "2000",
                "0": "100",
                "ZT": "70",
                "DT": "10",
                "qscln": "1200000000000",
            },
        })

    breadth, sentiment = fetch_historical_market_context(
        "2026-11-12",
        expected_market_count=5200,
        fetched_at="2026-11-13T08:00:00Z",
        post=fake_post,
    )

    assert calls[0][1]["a"] == "HisZhangFuDetail"
    assert calls[0][1]["c"] == "HisHomeDingPan"
    assert calls[0][1]["Day"] == "2026-11-12"
    assert calls[0][2] == 10
    assert breadth["trade_date"] == "2026-11-12"
    assert breadth["advances"] == 3000
    assert breadth["declines"] == 2000
    assert breadth["unchanged"] == 100
    assert breadth["breadth"] == pytest.approx(3000 / 5100)
    assert breadth["breadth_coverage"] == pytest.approx(5100 / 5200)
    assert breadth["total_amount"] == 1_200_000_000_000
    assert breadth["snapshot_time"] == "2026-11-12T07:40:00+00:00"
    assert breadth["available_time"] == "2026-11-13T08:00:00+00:00"
    assert sentiment["capture_time"] == breadth["snapshot_time"]
    assert sentiment["available_time"] == breadth["available_time"]
    assert sentiment["limit_up_count"] == 70
    assert sentiment["limit_down_count"] == 10
    assert breadth["historical_pit_eligible"] is False
    assert sentiment["historical_pit_eligible"] is False


def test_historical_market_context_rejects_incomplete_market_coverage():
    def fake_post(url, **kwargs):
        return FakeResponse({
            "errcode": "0",
            "info": {
                "SZJS": "2000",
                "XDJS": "1500",
                "0": "100",
                "ZT": "20",
                "DT": "5",
                "qscln": "1000000000000",
            },
        })

    with pytest.raises(ValueError, match="coverage below minimum"):
        fetch_historical_market_context(
            "2026-11-12",
            expected_market_count=5200,
            fetched_at="2026-11-13T08:00:00Z",
            post=fake_post,
        )


def test_historical_market_context_rejects_unavailable_historical_date():
    def fake_post(url, **kwargs):
        return FakeResponse({"errcode": "1001", "info": {}})

    with pytest.raises(RuntimeError, match="errcode"):
        fetch_historical_market_context(
            "2026-11-12",
            expected_market_count=5200,
            fetched_at="2026-11-13T08:00:00Z",
            post=fake_post,
        )
