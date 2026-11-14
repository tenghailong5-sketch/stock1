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


def test_download_retries_with_pinned_intermediate_bundle_after_ssl_error(monkeypatch):
    class SSLFailOnceSession:
        def __init__(self):
            self.calls = []

        def get(self, url, **kwargs):
            self.calls.append(kwargs)
            if len(self.calls) == 1:
                raise sw_industry.requests.exceptions.SSLError("untrusted local issuer")
            return Response(bytes.fromhex("D0CF11E0A1B11AE1") + b"xls")

    monkeypatch.setattr(
        sw_industry, "_trusted_intermediate_bundle",
        lambda timeout=15: "/tmp/pinned-sw-ca.pem",
    )
    session = SSLFailOnceSession()
    content = sw_industry._download_xls(session, sw_industry.STOCK_URL, 30)

    assert content.endswith(b"xls")
    assert len(session.calls) == 2
    assert session.calls[1]["verify"] == "/tmp/pinned-sw-ca.pem"


def test_download_rejects_html_error_page():
    class HtmlSession:
        def get(self, url, **kwargs):
            return Response(b"<html>blocked</html>")

    with pytest.raises(ValueError, match="not a valid OLE"):
        sw_industry._download_xls(HtmlSession(), sw_industry.STOCK_URL, 30)



def test_sw_history_provider_preserves_valid_time_intervals_without_claiming_pit(monkeypatch):
    captured = pd.Timestamp("2026-11-14T08:00:00Z")
    def fake_sheet_rows(content):
        if content.endswith(b"code-file"):
            return (["行业代码", "一级行业名称", "二级行业名称", "三级行业名称"],
                    iter([["110101", "农林牧渔", "种植业", "种子"],
                          ["220201", "基础化工", "化学原料", "纯碱"]]))
        return (["股票代码", "计入日期", "行业代码"],
                iter([["000001", 45000, "110101"], ["000001", 46000, "220201"],
                      ["600001", 45500, "110101"]]))
    monkeypatch.setattr(sw_industry, "_sheet_rows", fake_sheet_rows)
    result = sw_industry.fetch_sw_industry_valid_time_ledger(
        session=Session(), clock=lambda: captured)
    selected = result[result["symbol"].eq("000001")].sort_values("valid_from")
    assert len(selected) == 2
    assert selected.iloc[0]["industry_code"] == "110101"
    assert selected.iloc[1]["industry_code"] == "220201"
    assert selected.iloc[0]["valid_to"] == selected.iloc[1]["valid_from"]
    assert pd.isna(selected.iloc[1]["valid_to"])
    assert result["historical_pit_eligible"].eq(False).all()
    assert result["available_time"].eq(captured).all()
    assert result["snapshot_scope"].eq("historical_valid_time_intervals_not_pit").all()


def test_sw_history_provider_rejects_conflicting_same_day_memberships(monkeypatch):
    def fake_sheet_rows(content):
        if content.endswith(b"code-file"):
            return (["行业代码", "一级行业名称", "二级行业名称", "三级行业名称"],
                    iter([["110101", "农林牧渔", "种植业", "种子"],
                          ["220201", "基础化工", "化学原料", "纯碱"]]))
        return (["股票代码", "计入日期", "行业代码"],
                iter([["000001", 46000, "110101"], ["000001", 46000, "220201"]]))
    monkeypatch.setattr(sw_industry, "_sheet_rows", fake_sheet_rows)
    with pytest.raises(ValueError, match="conflicting industries"):
        sw_industry.fetch_sw_industry_valid_time_ledger(
            session=Session(), clock=lambda: pd.Timestamp("2026-11-14T08:00:00Z"))





def test_sw_history_unmapped_codes_are_audited_not_mapped(monkeypatch):
    def fake_sheet_rows(content):
        if content.endswith(b"code-file"):
            return (["行业代码", "一级行业名称", "二级行业名称", "三级行业名称"],
                    iter([["110101", "农林牧渔", "种植业", "种子"]]))
        return (["股票代码", "计入日期", "行业代码"],
                iter([["000001", 45000, "110101"], ["000001", 43000, "990001"],
                      ["600001", 44000, "990001"], ["000002", "", "990002"]]))
    monkeypatch.setattr(sw_industry, "_sheet_rows", fake_sheet_rows)
    result = sw_industry.fetch_sw_industry_unmapped_history_audit(
        session=Session(), current_symbols={"000001"}
    )
    assert result["industry_code"].tolist() == ["990001"]
    assert result.iloc[0]["row_count"] == 2
    assert result.iloc[0]["symbol_count"] == 2
    assert result.iloc[0]["current_member_symbol_count"] == 1
    assert result.iloc[0]["not_current_member_symbol_count"] == 1
    assert result.iloc[0]["mapping_status"] == "unmapped_outside_sw2021_dictionary"
    assert result.iloc[0]["mapping_precision"] == "unresolved_no_unique_sw2021_prefix"
    assert result.iloc[0]["industry_level3_candidate"] == ""



def test_sw_history_unmapped_audit_exposes_only_unique_prefix_candidates(monkeypatch):
    def fake_sheet_rows(content):
        if content.endswith(b"code-file"):
            return (["行业代码", "一级行业名称", "二级行业名称", "三级行业名称"],
                    iter([["110101", "农林牧渔", "种植业", "种子"],
                          ["110102", "农林牧渔", "种植业", "粮食种植"],
                          ["220201", "基础化工", "化学原料", "纯碱"]]))
        return (["股票代码", "计入日期", "行业代码"],
                iter([["000001", 45000, "110199"],
                      ["000002", 45001, "990001"]]))
    monkeypatch.setattr(sw_industry, "_sheet_rows", fake_sheet_rows)
    result = sw_industry.fetch_sw_industry_unmapped_history_audit(
        session=Session(), current_symbols={"000001"}
    ).set_index("industry_code")
    candidate = result.loc["110199"]
    assert candidate["industry_level1_candidate"] == "农林牧渔"
    assert candidate["industry_level2_candidate"] == "种植业"
    assert candidate["industry_level3_candidate"] == ""
    assert candidate["mapping_precision"] == "level2_prefix_candidate_unverified"
    unresolved = result.loc["990001"]
    assert unresolved["industry_level1_candidate"] == ""
    assert unresolved["industry_level2_candidate"] == ""
    assert unresolved["mapping_precision"] == "unresolved_no_unique_sw2021_prefix"
