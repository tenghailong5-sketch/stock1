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
