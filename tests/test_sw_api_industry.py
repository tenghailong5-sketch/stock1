import pandas as pd
import pytest

from src.providers.sw_api_industry import (
    fetch_sw_api_industry_membership_snapshot,
)


class FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class FakeSWSession:
    def __init__(self, fail_codes=None):
        self.fail_codes = set(fail_codes or [])
        self.calls = []
        self.verify = True

    def get(self, url, *, params, headers, timeout):
        self.calls.append((url, dict(params), timeout))
        if url.endswith("/current/"):
            return FakeResponse({
                "data": {
                    "count": 2,
                    "results": [
                        {"swindexcode": "801010", "swindexname": "农林牧渔"},
                        {"swindexcode": "801030", "swindexname": "基础化工"},
                    ],
                }
            })
        code = params["swindexcode"]
        if code in self.fail_codes:
            raise RuntimeError("provider group unavailable")
        members = {
            "801010": [{"stockcode": "000001", "stockname": "样例A"}],
            "801030": [{"stockcode": "600000", "stockname": "样例B"}],
        }
        return FakeResponse({
            "data": {
                "count": len(members[code]),
                "results": members[code],
            }
        })


def test_sw_api_provider_normalizes_members_and_keeps_taxonomy_lineage():
    session = FakeSWSession()
    captured = pd.Timestamp("2026-11-14T08:00:00Z")
    result = fetch_sw_api_industry_membership_snapshot(
        session=session,
        clock=lambda: captured,
        pause_seconds=0,
    )

    assert set(result["symbol"]) == {"000001", "600000"}
    assert set(result["industry_code"]) == {"801010", "801030"}
    assert result["source_id"].eq(
        "swsresearch.index_publish.component_stocks"
    ).all()
    assert result["source_taxonomy"].eq("SW").all()
    assert result["historical_pit_eligible"].eq(False).all()
    assert result["available_time"].eq(captured).all()
    assert result["classification_level"].eq("一级行业").all()
    assert result.attrs["requested_industry_count"] == 2
    assert result.attrs["failed_industry_codes"] == []


def test_sw_api_provider_reports_partial_group_failures_without_losing_good_rows():
    session = FakeSWSession(fail_codes={"801030"})
    result = fetch_sw_api_industry_membership_snapshot(
        session=session,
        clock=lambda: pd.Timestamp("2026-11-14T08:00:00Z"),
        pause_seconds=0,
    )

    assert set(result["symbol"]) == {"000001"}
    assert result.attrs["failed_industry_codes"] == ["801030"]
    assert result.attrs["requested_industry_count"] == 2


def test_sw_api_provider_rejects_unknown_industry_level():
    with pytest.raises(ValueError, match="industry_level"):
        fetch_sw_api_industry_membership_snapshot(
            session=FakeSWSession(), industry_level="概念"
        )


def test_sw_api_provider_rejects_empty_industry_group_response():
    class EmptySession(FakeSWSession):
        def get(self, url, *, params, headers, timeout):
            if url.endswith("/current/"):
                return FakeResponse({"data": {"count": 0, "results": []}})
            return super().get(url, params=params, headers=headers, timeout=timeout)

    with pytest.raises(RuntimeError, match="no industry index groups"):
        fetch_sw_api_industry_membership_snapshot(
            session=EmptySession(), pause_seconds=0
        )
