import pandas as pd
import pytest

from src.providers.sw_industry_history import (
    fetch_sw_industry_history,
    normalize_sw_industry_history,
)


def source_rows():
    return pd.DataFrame([
        {"股票代码": "1", "计入日期": "1991-04-03", "行业代码": "440101", "更新日期": "2021-07-30"},
        {"股票代码": "1", "计入日期": "2014-02-21", "行业代码": "480101", "更新日期": "2021-07-30"},
        {"股票代码": "1", "计入日期": "2021-07-30", "行业代码": "480301", "更新日期": "2021-07-30"},
        {"股票代码": "600000", "计入日期": "2000-01-01", "行业代码": "480201", "更新日期": None},
    ])


def test_builds_effective_date_intervals_and_industry_hierarchy():
    result = normalize_sw_industry_history(
        source_rows(), fetched_at="2026-11-15T00:00:00Z"
    )
    stock = result[result.symbol == "000001"].reset_index(drop=True)
    assert stock.industry_code.tolist() == ["440101", "480101", "480301"]
    assert stock.l1_code.tolist() == ["440000", "480000", "480000"]
    assert stock.l2_code.tolist() == ["440100", "480100", "480300"]
    assert stock.valid_to.iloc[0] == pd.Timestamp("2014-02-21")
    assert stock.valid_to.iloc[1] == pd.Timestamp("2021-07-30")
    assert pd.isna(stock.valid_to.iloc[2])


def test_retrieval_time_is_not_promoted_to_historical_pit():
    result = normalize_sw_industry_history(
        source_rows(), fetched_at="2026-11-15T00:00:00Z"
    )
    assert result.availability_basis.eq("retrieval_time_only").all()
    assert result.historical_pit_eligible.eq(False).all()
    assert result.available_time.dt.tz is not None


def test_numeric_excel_codes_are_normalized_without_decimal_suffix():
    frame = source_rows().iloc[:1].copy()
    frame["股票代码"] = 1.0
    frame["行业代码"] = 440101.0
    result = normalize_sw_industry_history(frame)
    assert result.symbol.iloc[0] == "000001"
    assert result.industry_code.iloc[0] == "440101"


def test_missing_required_source_column_fails_closed():
    with pytest.raises(ValueError, match="missing columns"):
        normalize_sw_industry_history(source_rows().drop(columns=["行业代码"]))


def test_invalid_effective_date_fails_closed():
    frame = source_rows().iloc[:1].copy()
    frame.loc[0, "计入日期"] = "not-a-date"
    with pytest.raises(ValueError, match="invalid effective date"):
        normalize_sw_industry_history(frame)


def test_duplicate_stock_effective_date_fails_closed():
    frame = pd.concat([source_rows(), source_rows().iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate symbol/effective date"):
        normalize_sw_industry_history(frame)


def test_naive_capture_time_is_rejected():
    with pytest.raises(ValueError, match="timezone-aware"):
        normalize_sw_industry_history(
            source_rows(), fetched_at="2026-11-15 00:00:00"
        )


def test_fetch_uses_official_workbook_and_normalizes_response():
    class Response:
        content = b"fake-xls"
        def raise_for_status(self):
            return None

    calls = []
    def fake_get(url, *, headers, timeout):
        calls.append((url, headers, timeout))
        return Response()

    def fake_read_excel(_):
        return source_rows()

    import src.providers.sw_industry_history as provider
    original = provider.pd.read_excel
    provider.pd.read_excel = fake_read_excel
    try:
        result = fetch_sw_industry_history(
            http_get=fake_get, fetched_at="2026-11-15T00:00:00Z"
        )
    finally:
        provider.pd.read_excel = original

    assert len(calls) == 1
    assert "swsresearch.com" in calls[0][0]
    assert set(result.symbol) == {"000001", "600000"}
    assert result.historical_pit_eligible.eq(False).all()


def test_fetch_failure_is_reported_not_returned_as_empty_data():
    def fail_get(*args, **kwargs):
        raise ConnectionError("offline")
    with pytest.raises(RuntimeError, match="fetch/parse failed"):
        fetch_sw_industry_history(http_get=fail_get)
