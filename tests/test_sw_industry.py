import pandas as pd
import pytest

from src.providers import sw_industry


class Response:
    def __init__(self, content):
        self.content = content

    def raise_for_status(self):
        return None


class Session:
    def __init__(self):
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append(url)
        return Response(bytes.fromhex("D0CF11E0A1B11AE1") + (b"stock-file" if "StockClassify" in url else b"code-file"))


def test_sw_provider_selects_latest_valid_classification_and_stamps_retrieval(monkeypatch):
    def fake_sheet_rows(content):
        if content.endswith(b"code-file"):
            return (
                ["行业代码", "一级行业名称", "二级行业名称", "三级行业名称"],
                iter([
                    ["110101", "农林牧渔", "种植业", "种子"],
                    ["220201", "基础化工", "化学原料", "纯碱"],
                ]),
            )
        return (
            ["股票代码", "计入日期", "行业代码"],
            iter([
                ["000001", 45000, "110101"],
                ["000001", 46000, "220201"],
                ["600001", 45500, "110101"],
                ["000002", 46000, "999999"],  # old/unmapped classification is not fabricated
            ]),
        )

    monkeypatch.setattr(sw_industry, "_sheet_rows", fake_sheet_rows)
    session = Session()
    captured = pd.Timestamp("2026-11-13T14:00:00Z")
    result = sw_industry.fetch_sw_current_industry_membership_snapshot(
        session=session, clock=lambda: captured
    )

    assert session.calls == [sw_industry.STOCK_URL, sw_industry.CODE_URL]
    assert set(result["symbol"]) == {"000001", "600001"}
    selected = result.set_index("symbol")
    assert selected.loc["000001", "industry_code"] == "220201"
    assert selected.loc["000001", "industry_name"] == "纯碱"
    assert selected.loc["000001", "industry_level1"] == "基础化工"
    assert selected.loc["000001", "observed_at"] == captured
    assert set(result["source_id"]) == {"swsresearch.sw2021_stock_classification"}
    assert result["historical_pit_eligible"].eq(False).all()


def test_sw_provider_rejects_naive_observation_clock(monkeypatch):
    def fake_sheet_rows(content):
        if content.endswith(b"code-file"):
            return (
                ["行业代码", "一级行业名称", "二级行业名称", "三级行业名称"],
                iter([["110101", "农林牧渔", "种植业", "种子"]]),
            )
        return (
            ["股票代码", "计入日期", "行业代码"],
            iter([["000001", 46000, "110101"]]),
        )

    monkeypatch.setattr(sw_industry, "_sheet_rows", fake_sheet_rows)
    with pytest.raises(ValueError, match="timezone-aware"):
        sw_industry.fetch_sw_current_industry_membership_snapshot(
            session=Session(), clock=lambda: pd.Timestamp("2026-11-13T14:00:00")
        )


def test_download_retries_with_system_ca_after_ssl_error(monkeypatch):
    class SSLFailOnceSession:
        def __init__(self):
            self.calls = []

        def get(self, url, **kwargs):
            self.calls.append(kwargs)
            if len(self.calls) == 1:
                raise sw_industry.requests.exceptions.SSLError("untrusted local issuer")
            return Response(bytes.fromhex("D0CF11E0A1B11AE1") + b"xls")

    monkeypatch.setattr(sw_industry.os.path, "isfile", lambda path: True)
    session = SSLFailOnceSession()
    content = sw_industry._download_xls(session, sw_industry.STOCK_URL, 30)

    assert content.endswith(b"xls")
    assert len(session.calls) == 2
    assert session.calls[1]["verify"] == sw_industry.SYSTEM_CA_BUNDLE


def test_download_rejects_html_error_page():
    class HtmlSession:
        def get(self, url, **kwargs):
            return Response(b"<html>blocked</html>")

    with pytest.raises(ValueError, match="not a valid OLE"):
        sw_industry._download_xls(HtmlSession(), sw_industry.STOCK_URL, 30)
