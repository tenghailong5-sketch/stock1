import pandas as pd
import pytest

from src.providers.baidu_industry import fetch_baidu_industry_membership_snapshot


class Response:
    def __init__(self, data):
        self.data = data

    def raise_for_status(self):
        return None

    def json(self):
        return self.data


class Session:
    def __init__(self):
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append(kwargs)
        return Response({
            "Result": {
                "000001": [{
                    "name": "行业",
                    "list": [{
                        "name": "银行",
                        "describe": "银行",
                        "xcx_query": "code=801780"
                    }]
                }],
                "600000": [{
                    "name": "行业",
                    "list": [{
                        "name": "股份制银行",
                        "describe": "银行",
                        "xcx_query": "code=801781"
                    }]
                }]
            }
        })


def test_baidu_industry_maps_each_requested_security_and_preserves_source():
    session = Session()
    captured = pd.Timestamp("2026-11-14T00:00:00Z")
    result = fetch_baidu_industry_membership_snapshot(
        ["000001", "600000"], session=session, clock=lambda: captured,
        pause_seconds=0,
    )
    assert set(result["symbol"]) == {"000001", "600000"}
    assert set(result["source_id"]) == {"baidu.gushitong.relatedblock"}
    assert result["historical_pit_eligible"].eq(False).all()
    assert result["observed_at"].eq(captured).all()
    assert result.set_index("symbol").loc["000001", "industry_code"] == "801780"


def test_baidu_industry_rejects_empty_universe():
    with pytest.raises(ValueError, match="universe is empty"):
        fetch_baidu_industry_membership_snapshot([])


def test_baidu_industry_rejects_missing_result():
    class BadSession:
        def get(self, url, **kwargs):
            return Response({"error": "blocked"})

    with pytest.raises(ValueError, match="no Result"):
        fetch_baidu_industry_membership_snapshot(
            ["000001"], session=BadSession(), pause_seconds=0
        )
