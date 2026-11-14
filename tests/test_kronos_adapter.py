import numpy as np
import pandas as pd
import pytest

from src.model.kronos_adapter import (
    KronosAdapter, KronosConfig, prepare_kronos_history, summarize_forecast,
)


def _history(n=40):
    dates = pd.date_range("2026-09-01", periods=n, freq="B")
    close = np.linspace(10.0, 12.0, n)
    return pd.DataFrame({
        "timestamp": dates,
        "available_time": [pd.Timestamp(d).tz_localize("Asia/Shanghai") + pd.Timedelta(hours=16) for d in dates],
        "open": close - 0.05,
        "high": close + 0.10,
        "low": close - 0.10,
        "close": close,
        "volume": np.full(n, 100000.0),
        "amount": close * 100000.0,
    })


class FakePredictor:
    def __init__(self):
        self.kwargs = None

    def predict(self, **kwargs):
        self.kwargs = kwargs
        dates = pd.DatetimeIndex(kwargs["y_timestamp"])
        last = float(kwargs["df"]["close"].iloc[-1])
        close = [last * 1.01, last * 1.02, last * 1.015]
        return pd.DataFrame({
            "open": close,
            "high": [v * 1.01 for v in close],
            "low": [v * 0.99 for v in close],
            "close": close,
            "volume": [100000.0] * len(close),
            "amount": [v * 100000.0 for v in close],
        }, index=dates)


def test_prepare_history_excludes_late_and_future_rows():
    frame = _history()
    late = frame.iloc[[-1]].copy()
    late["timestamp"] = pd.Timestamp("2026-12-01")
    late["available_time"] = pd.Timestamp("2026-12-01T16:00:00+08:00")
    frame = pd.concat([frame, late], ignore_index=True)
    cutoff = pd.Timestamp("2026-10-27T15:00:00+08:00")
    model_frame, stamps, audit = prepare_kronos_history(
        frame, decision_time=cutoff, lookback=256,
    )
    assert len(model_frame) < len(frame)
    assert stamps.max() <= cutoff.tz_convert("Asia/Shanghai").tz_localize(None)
    assert audit["availability_rule"] == "available_time<=decision_time"


def test_prepare_history_rejects_unzoned_availability_and_invalid_ohlc():
    frame = _history()
    frame["available_time"] = pd.to_datetime(frame["timestamp"])
    with pytest.raises(ValueError, match="timezone-aware"):
        prepare_kronos_history(frame, decision_time="2026-10-27T16:00:00+08:00")

    frame = _history()
    frame.loc[0, "high"] = frame.loc[0, "low"] - 1
    with pytest.raises(ValueError, match="OHLC"):
        prepare_kronos_history(frame, decision_time="2026-10-27T16:00:00+08:00")


def test_forecast_uses_only_future_sessions_supplied_by_caller():
    predictor = FakePredictor()
    adapter = KronosAdapter(predictor, KronosConfig(lookback=32))
    history = _history()
    decision_time = pd.Timestamp("2026-10-27T17:00:00+08:00")
    future = pd.date_range("2026-10-28", periods=3, freq="B")
    result = adapter.forecast(
        history, decision_time=decision_time, future_timestamps=future,
    )
    assert result["status"] == "READY"
    assert result["historical_pit_eligible"] is False
    assert result["metrics"]["forecast_return_1"] > 0
    assert result["metrics"]["probability"] is None
    assert result["metrics"]["trading_score"] is None
    assert predictor.kwargs["pred_len"] == 3
    assert predictor.kwargs["verbose"] is False


def test_forecast_rejects_non_future_or_unsorted_session_dates():
    adapter = KronosAdapter(FakePredictor(), KronosConfig(lookback=32))
    history = _history()
    with pytest.raises(ValueError, match="later than the last input bar"):
        adapter.forecast(
            history, decision_time="2026-10-27T17:00:00+08:00",
            future_timestamps=[pd.Timestamp("2026-10-01")],
        )
    with pytest.raises(ValueError, match="unique and increasing"):
        adapter.forecast(
            history, decision_time="2026-10-27T17:00:00+08:00",
            future_timestamps=[pd.Timestamp("2026-10-30"), pd.Timestamp("2026-10-29")],
        )


def test_config_caps_small_context_and_validates_sampling():
    with pytest.raises(ValueError, match="between 32 and 512"):
        KronosConfig(lookback=600)
    with pytest.raises(ValueError, match="sample_count"):
        KronosConfig(sample_count=0)


def test_summary_does_not_claim_probabilities_or_trade_scores():
    prediction = pd.DataFrame({
        "open": [10.1, 10.2, 10.3], "high": [10.3, 10.4, 10.5],
        "low": [9.9, 10.0, 10.1], "close": [10.2, 10.3, 10.4],
        "volume": [1, 1, 1], "amount": [10, 10, 10],
    })
    metrics = summarize_forecast(prediction, last_close=10.0)
    assert metrics["forecast_return_3"] == pytest.approx(0.04)
    assert metrics["probability"] is None
    assert metrics["trading_score"] is None
