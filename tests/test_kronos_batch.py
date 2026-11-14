import numpy as np
import pandas as pd
import pytest

from src.model.kronos_adapter import KronosAdapter, KronosConfig
from src.model.kronos_batch import forecast_candidate_pool, normalize_candidate_history


def _history(n=40):
    dates = pd.date_range("2026-09-01", periods=n, freq="B")
    close = np.linspace(10.0, 12.0, n)
    return pd.DataFrame({
        "date": dates,
        "available_time": [
            pd.Timestamp(d).tz_localize("Asia/Shanghai") + pd.Timedelta(hours=16)
            for d in dates
        ],
        "open": close - 0.05,
        "high": close + 0.10,
        "low": close - 0.10,
        "close": close,
        "volume": np.full(n, 100000.0),
        "amount": close * 100000.0,
    })


class FakePredictor:
    def predict(self, **kwargs):
        last = float(kwargs["df"]["close"].iloc[-1])
        close = [last * (1 + 0.01 * (i + 1)) for i in range(kwargs["pred_len"])]
        return pd.DataFrame({
            "open": close,
            "high": [v * 1.01 for v in close],
            "low": [v * 0.99 for v in close],
            "close": close,
            "volume": [100000.0] * len(close),
            "amount": [v * 100000.0 for v in close],
        })


def test_normalize_fhm_history_schema():
    normalized = normalize_candidate_history(_history())
    assert set(normalized.columns) == {
        "timestamp", "available_time", "open", "high", "low", "close",
        "volume", "amount",
    }


def test_batch_reuses_one_injected_predictor_and_returns_per_symbol_metrics():
    adapter = KronosAdapter(FakePredictor(), KronosConfig(lookback=32))
    histories = {"000001": _history(), "600000": _history()}
    result = forecast_candidate_pool(
        histories,
        decision_time=pd.Timestamp("2026-10-27T17:00:00+08:00"),
        future_timestamps=pd.date_range("2026-10-28", periods=5, freq="B"),
        adapter=adapter,
    )
    assert result["status"] == "READY"
    assert result["forecast_count"] == 2
    assert result["error_count"] == 0
    assert result["results"]["000001"]["metrics"]["forecast_return_5"] > 0
    assert result["historical_pit_eligible"] is False


def test_batch_is_bounded_and_reports_bad_symbol_without_dropping_good_names():
    adapter = KronosAdapter(FakePredictor(), KronosConfig(lookback=32))
    bad = _history().drop(columns=["amount"])
    result = forecast_candidate_pool(
        {"000001": _history(), "BAD": bad},
        decision_time="2026-10-27T17:00:00+08:00",
        future_timestamps=pd.date_range("2026-10-28", periods=3, freq="B"),
        adapter=adapter,
    )
    assert result["forecast_count"] == 1
    assert result["error_count"] == 1
    assert "missing" in result["errors"]["BAD"]


def test_batch_rejects_pool_over_limit_before_model_loading():
    with pytest.raises(ValueError, match="exceeds configured cap"):
        forecast_candidate_pool(
            {str(i): _history() for i in range(3)},
            decision_time="2026-10-27T17:00:00+08:00",
            future_timestamps=pd.date_range("2026-10-28", periods=3, freq="B"),
            adapter=KronosAdapter(FakePredictor(), KronosConfig(lookback=32)),
            max_candidates=2,
        )
