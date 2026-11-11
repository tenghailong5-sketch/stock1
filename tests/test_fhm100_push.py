import pytest

from src import fhm100


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


def test_pushplus_rejection_raises(monkeypatch):
    monkeypatch.setenv("PUSHPLUS_TOKEN", "test-token")
    monkeypatch.setattr(
        fhm100.requests,
        "post",
        lambda *args, **kwargs: FakeResponse({"code": 500, "msg": "rejected"}),
    )

    with pytest.raises(RuntimeError, match="PushPlus rejected"):
        fhm100.push("test report")


def test_pushplus_success_requires_code_200(monkeypatch):
    monkeypatch.setenv("PUSHPLUS_TOKEN", "test-token")
    monkeypatch.setattr(
        fhm100.requests,
        "post",
        lambda *args, **kwargs: FakeResponse({"code": 200, "msg": "success"}),
    )

    result = fhm100.push("test report")

    assert "code=200" in result


def test_pushplus_missing_token_is_explicit_noop(monkeypatch):
    monkeypatch.delenv("PUSHPLUS_TOKEN", raising=False)

    result = fhm100.push("test report")

    assert "未配置PUSHPLUS_TOKEN" in result


def test_history_retries_transient_timeout_then_succeeds(monkeypatch):
    calls = {"count": 0}
    expected = object()

    def flaky_request(**kwargs):
        calls["count"] += 1
        if calls["count"] < 3:
            raise fhm100.requests.exceptions.ReadTimeout("temporary timeout")
        return expected

    monkeypatch.setattr(fhm100.ak, "stock_zh_a_hist", flaky_request)
    monkeypatch.setattr(fhm100.time, "sleep", lambda seconds: None)

    assert fhm100._download_history("000001", "20260101") is expected
    assert calls["count"] == 3


def test_history_does_not_retry_non_transport_errors(monkeypatch):
    calls = {"count": 0}

    def invalid_request(**kwargs):
        calls["count"] += 1
        raise ValueError("schema/data error")

    monkeypatch.setattr(fhm100.ak, "stock_zh_a_hist", invalid_request)
    monkeypatch.setattr(fhm100.time, "sleep", lambda seconds: None)

    with pytest.raises(ValueError, match="schema/data error"):
        fhm100._download_history("000001", "20260101")
    assert calls["count"] == 1


def test_history_rejects_zero_retry_attempts():
    with pytest.raises(ValueError, match="max_attempts"):
        fhm100._download_history("000001", "20260101", max_attempts=0)
