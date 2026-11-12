import pandas as pd
import pytest

from scripts import backfill_tencent_history as backfill


def test_normalizes_exchange_symbols():
    assert backfill.norm_code("600000") == "sh600000"
    assert backfill.norm_code("000001") == "sz000001"
    assert backfill.norm_code("430047") == "bj430047"
    assert backfill.norm_code("not-a-code") is None


def test_normalizes_exchange_security_list():
    frame = pd.DataFrame({"证券代码": ["600000", "000001"], "证券简称": ["浦发银行", "平安银行"]})
    result = backfill.norm_list(frame, "test")
    assert result["symbol"].tolist() == ["sh600000", "sz000001"]


def test_fetch_bars_validates_and_normalizes_tencent_daily_bars(monkeypatch):
    raw = pd.DataFrame([
        {"date": "2026-11-11", "open": "10", "high": "11", "low": "9", "close": "10.5", "volume": "1000", "amount": "10500"},
        {"date": "2026-11-12", "open": "10.5", "high": "12", "low": "10", "close": "11.5", "volume": "1200", "amount": "13800"},
    ])
    monkeypatch.setattr(backfill.ak, "stock_zh_a_hist_tx", lambda **kwargs: raw)
    result = backfill.fetch_bars("sz000001", "20261101", "20261112")
    assert result["symbol"].unique().tolist() == ["sz000001"]
    assert result["source"].unique().tolist() == ["tencent"]
    assert result["amount"].tolist() == [10500, 13800]


def test_fetch_bars_rejects_missing_amount(monkeypatch):
    raw = pd.DataFrame([
        {"date": "2026-11-11", "open": 10, "high": 11, "low": 9, "close": 10.5, "volume": 1000, "amount": None}
    ])
    monkeypatch.setattr(backfill.ak, "stock_zh_a_hist_tx", lambda **kwargs: raw)
    with pytest.raises(ValueError, match="null/non-finite OHLCVA"):
        backfill.fetch_bars("sz000001", "20261101", "20261112")
