import pandas as pd
import requests

from src import fhm100


def _provider_history(rows=30):
    dates = pd.bdate_range("2026-09-01", periods=rows)
    close = [10.0 + i * 0.03 for i in range(rows)]
    return pd.DataFrame({
        "日期": dates.strftime("%Y-%m-%d"),
        "开盘": [value - 0.02 for value in close],
        "收盘": close,
        "最高": [value + 0.05 for value in close],
        "最低": [value - 0.05 for value in close],
        "成交量": [1_000_000] * rows,
        "成交额": [200_000_000] * rows,
        "换手率": [1.0] * rows,
    })


def _normalized_history(rows=30):
    raw = _provider_history(rows)
    captured = pd.Timestamp("2026-11-11T08:00:00Z")
    lineage = fhm100.normalize_daily_history_snapshot(
        raw, symbol="000001", fetched_at=captured
    )[["trade_date", "available_time", "source_id", "source_revision"]]
    frame = raw.rename(columns={
        "日期": "date", "开盘": "open", "收盘": "close",
        "最高": "high", "最低": "low", "成交量": "volume",
        "成交额": "amount",
    }).copy()
    frame["date"] = pd.to_datetime(frame["date"])
    return frame.merge(lineage, left_on="date", right_on="trade_date").drop(columns="trade_date")


def test_history_attaches_row_level_provider_lineage(monkeypatch, tmp_path):
    raw = _provider_history()
    monkeypatch.setattr(fhm100, "_download_history", lambda *args, **kwargs: raw)
    monkeypatch.setattr(fhm100, "archive_stock_history_snapshot", lambda *args, **kwargs: {
        "rows_appended": len(raw), "manifests_appended": 1,
    })

    result = fhm100.history("000001", "20260901", archive_root=tmp_path)

    assert len(result) == len(raw)
    assert {"available_time", "source_id", "source_revision"}.issubset(result.columns)
    assert result["source_id"].eq("akshare.stock_zh_a_hist").all()
    assert result["source_revision"].str.len().eq(64).all()
    assert result["available_time"].dt.tz is not None
    assert not result["available_time"].dt.tz_convert("UTC").isna().any()


def test_scan_passes_lineage_into_m4_instead_of_dropping_every_symbol(monkeypatch, tmp_path):
    history = _normalized_history()
    monkeypatch.setattr(fhm100, "history", lambda *args, **kwargs: history.copy())
    universe = pd.DataFrame([{
        "code": "000001", "name": "测试股票", "price": 10.87,
        "amount": 200_000_000.0, "pct_chg": 0.5,
    }])

    watchlist, errors = fhm100.scan(universe, archive_root=tmp_path)

    assert errors.empty
    assert len(watchlist) == 1
    assert watchlist.iloc[0]["代码"] == "000001"
    assert watchlist.iloc[0]["M4跨日价格状态"] == "MISSING_PIT_TOTAL_RETURN_FIELDS"
    assert "本次抓取时间戳不具备历史PIT资格" in watchlist.iloc[0]["M4数据口径"]

    
def test_universe_uses_sina_only_after_tencent_fallback_fails(monkeypatch):
    class Push2RateLimitError(Exception):
        pass

    spot = pd.DataFrame({
        "代码": ["000001"], "名称": ["平安银行"], "最新价": [10.0],
        "成交额": [300_000_000], "涨跌幅": [1.2],
    })
    monkeypatch.setattr(fhm100.ak, "stock_zh_a_spot_em",
                        lambda: (_ for _ in ()).throw(Push2RateLimitError("cooldown")))
    monkeypatch.setattr(fhm100.ak, "stock_zh_a_spot_tx",
                        lambda: (_ for _ in ()).throw(requests.exceptions.Timeout("Tencent timeout")))
    monkeypatch.setattr(fhm100.ak, "stock_zh_a_spot", lambda: spot.copy())

    result = fhm100.universe()

    assert result["code"].iloc[0] == "000001"
    assert result.iloc[0]["amount"] == 300_000_000


def test_universe_prefers_tencent_snapshot_and_converts_turnover_to_yuan(monkeypatch):
    class Push2RateLimitError(Exception):
        pass

    rows = [{
        "code": f"sz{i:06d}", "name": f"普通股票{i}", "zxj": 10.0,
        "zdf": 0.1, "volume": 1_000_000, "turnover": 1_000,
    } for i in range(1, 1001)]
    rows[0].update({"code": "sz000001", "name": "平安银行", "zxj": 10.0,
                    "zdf": 1.2, "turnover": 30_000})
    rows[1].update({"code": "sh600000", "name": "浦发银行", "zxj": 9.0,
                    "zdf": -0.5, "turnover": 20_000})
    tencent = pd.DataFrame(rows)
    monkeypatch.setattr(fhm100.ak, "stock_zh_a_spot_em",
                        lambda: (_ for _ in ()).throw(Push2RateLimitError("cooldown")))
    monkeypatch.setattr(fhm100.ak, "stock_zh_a_spot_tx", lambda: tencent.copy())
    monkeypatch.setattr(fhm100.ak, "stock_zh_a_spot",
                        lambda: (_ for _ in ()).throw(AssertionError("Sina must not be called")))

    result = fhm100.universe()

    assert result["code"].head(2).tolist() == ["000001", "600000"]
    assert result.set_index("code").loc["000001", "amount"] == 300_000_000
    assert result.set_index("code").loc["600000", "amount"] == 200_000_000


def test_universe_falls_back_to_tencent_on_eastmoney_timeout(monkeypatch):
    rows = [{
        "code": f"sz{i:06d}", "name": f"普通股票{i}", "zxj": 10.0,
        "zdf": 0.1, "volume": 1_000_000, "turnover": 1_000,
    } for i in range(1, 1001)]
    rows[0].update({"code": "sz000001", "name": "平安银行", "zxj": 10.0,
                    "zdf": 1.2, "turnover": 30_000})
    tencent = pd.DataFrame(rows)
    monkeypatch.setattr(fhm100.ak, "stock_zh_a_spot_em",
                        lambda: (_ for _ in ()).throw(requests.exceptions.ReadTimeout("timeout")))
    monkeypatch.setattr(fhm100.ak, "stock_zh_a_spot_tx", lambda: tencent.copy())
    result = fhm100.universe()
    assert result["code"].iloc[0] == "000001"
    assert result.iloc[0]["amount"] == 300_000_000


def test_tencent_history_fallback_is_normalized_and_keeps_source_lineage(monkeypatch):
    class Push2RateLimitError(Exception):
        pass

    tencent = pd.DataFrame({
        "date": pd.bdate_range("2026-09-01", periods=30).strftime("%Y-%m-%d"),
        "open": [10.0] * 30, "high": [10.2] * 30, "low": [9.8] * 30,
        "close": [10.1] * 30, "volume": [1_000_000] * 30,
        "amount": [200_000_000] * 30,
    })
    monkeypatch.setattr(fhm100.ak, "stock_zh_a_hist",
                        lambda **kwargs: (_ for _ in ()).throw(Push2RateLimitError("cooldown")))
    monkeypatch.setattr(fhm100.ak, "stock_zh_a_hist_tx", lambda **kwargs: tencent.copy())
    result = fhm100._download_history("000001", "20260901")

    assert {"日期", "开盘", "最高", "最低", "收盘", "成交量", "成交额", "换手率"}.issubset(result.columns)
    assert result.attrs["source_id"] == "akshare.stock_zh_a_hist_tx"


def test_empty_scan_result_contains_columns_required_by_report_sort():
    watchlist, errors = fhm100.empty_scan_result(RuntimeError("provider unavailable"))

    assert watchlist.empty
    assert {"技术观察通过", "流动性观察通过", "当日成交额"}.issubset(watchlist.columns)
    assert watchlist.columns.is_unique
    # The exact final-report sort must remain safe when the live universe is unavailable.
    sorted_watchlist = watchlist.sort_values(
        ["技术观察通过", "流动性观察通过", "当日成交额"],
        ascending=[False, False, False],
    )
    assert sorted_watchlist.empty


def test_all_failed_history_scan_gets_sortable_empty_watchlist_schema():
    # scan() can return zero valid rows without raising, while preserving per-stock errors.
    raw_empty = pd.DataFrame()
    watchlist = fhm100.ensure_watchlist_schema(raw_empty)
    assert watchlist.empty
    assert {"技术观察通过", "流动性观察通过", "当日成交额"}.issubset(watchlist.columns)
    assert watchlist.columns.is_unique
    assert watchlist.sort_values(
        ["技术观察通过", "流动性观察通过", "当日成交额"],
        ascending=[False, False, False],
    ).empty
