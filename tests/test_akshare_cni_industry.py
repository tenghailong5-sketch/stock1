import pandas as pd
import pytest

from src.providers.akshare_cni_industry import (
    fetch_cni_historical_industry,
    normalize_cni_historical_industry,
)


def source_rows():
    return pd.DataFrame([
        {"日期": "2024-06-28", "样本代码": "000001", "所属行业": "金融"},
        {"日期": "2024-06-28", "样本代码": "000002", "所属行业": "房地产"},
        {"日期": "2024-12-31", "样本代码": "000001", "所属行业": "金融"},
    ])


def test_normalizes_historical_snapshots_without_fabricating_pit_time():
    result = normalize_cni_historical_industry(
        source_rows(),
        index_symbol="399001",
        fetched_at="2026-11-14T08:00:00+08:00",
    )
    assert len(result) == 3
    assert set(result["index_symbol"]) == {"399001"}
    assert set(result["industry_scheme"]) == {"CNI"}
    assert set(result["industry_code"]) == {"CNI:金融", "CNI:房地产"}
    assert result["membership_date"].max() < result["available_time"].min().tz_localize(None)
    assert result["available_time"].dt.tz is not None
    assert result["availability_basis"].eq("retrieval_time_only").all()
    assert result["historical_pit_eligible"].eq(False).all()
    assert result["membership_scope"].eq("index_constituents_only").all()


def test_missing_industry_is_rejected_not_dropped():
    frame = source_rows()
    frame.loc[0, "所属行业"] = None
    with pytest.raises(ValueError, match="missing industry"):
        normalize_cni_historical_industry(frame, index_symbol="399001")


def test_duplicate_symbol_date_is_rejected():
    frame = pd.concat([source_rows(), source_rows().iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate symbol/date"):
        normalize_cni_historical_industry(frame, index_symbol="399001")


def test_naive_fetch_time_is_rejected():
    with pytest.raises(ValueError, match="timezone-aware"):
        normalize_cni_historical_industry(
            source_rows(), index_symbol="399001", fetched_at="2026-11-14 08:00:00"
        )


def test_invalid_index_symbol_is_rejected():
    with pytest.raises(ValueError, match="numeric index code"):
        normalize_cni_historical_industry(source_rows(), index_symbol="CSI300")


def test_fetch_calls_each_explicit_index_and_preserves_scope():
    class FakeAk:
        calls = []

        @classmethod
        def index_detail_hist_cni(cls, *, symbol):
            cls.calls.append(symbol)
            return source_rows()

    result = fetch_cni_historical_industry(
        ["399001", "399005"],
        ak_module=FakeAk,
        fetched_at="2026-11-14T00:00:00Z",
    )
    assert FakeAk.calls == ["399001", "399005"]
    assert set(result["index_symbol"]) == {"399001", "399005"}
    assert result["historical_pit_eligible"].eq(False).all()


def test_fetch_fails_closed_if_any_index_fails():
    class PartialAk:
        @staticmethod
        def index_detail_hist_cni(*, symbol):
            if symbol == "399005":
                raise ConnectionError("offline")
            return source_rows()

    with pytest.raises(RuntimeError, match="399005"):
        fetch_cni_historical_industry(
            ["399001", "399005"],
            ak_module=PartialAk,
            fetched_at="2026-11-14T00:00:00Z",
        )


def test_duplicate_index_requests_are_rejected():
    with pytest.raises(ValueError, match="duplicates"):
        fetch_cni_historical_industry(["399001", "399001"])
