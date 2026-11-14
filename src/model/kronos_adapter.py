"""Kronos inference adapter with strict decision-time data boundaries.

This module deliberately does not assign trading scores or authorize entries.
The optional upstream Kronos package is imported only when a model is loaded.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence
import sys

import numpy as np
import pandas as pd


PRICE_COLUMNS = ("open", "high", "low", "close")
MODEL_COLUMNS = (*PRICE_COLUMNS, "volume", "amount")
MAX_SMALL_CONTEXT = 512


@dataclass(frozen=True)
class KronosConfig:
    model_name: str = "NeoQuasar/Kronos-small"
    tokenizer_name: str = "NeoQuasar/Kronos-Tokenizer-base"
    device: str = "cpu"
    repo_path: str | None = None
    lookback: int = 256
    sample_count: int = 1
    temperature: float = 1.0
    top_p: float = 0.9

    def __post_init__(self) -> None:
        if not 32 <= self.lookback <= MAX_SMALL_CONTEXT:
            raise ValueError("lookback must be between 32 and 512")
        if self.sample_count < 1:
            raise ValueError("sample_count must be positive")
        if self.temperature <= 0 or not 0 < self.top_p <= 1:
            raise ValueError("invalid sampling parameters")


def _market_local_naive(values: Any) -> pd.DatetimeIndex:
    parsed = pd.DatetimeIndex(pd.to_datetime(values, errors="coerce"))
    if parsed.isna().any():
        raise ValueError("timestamps contain invalid values")
    if parsed.tz is not None:
        parsed = parsed.tz_convert("Asia/Shanghai").tz_localize(None)
    return parsed


def prepare_kronos_history(
    frame: pd.DataFrame,
    *,
    decision_time: Any,
    lookback: int = 256,
) -> tuple[pd.DataFrame, pd.Series, dict]:
    """Return only complete bars known by decision_time, in market-local time."""
    required = {"timestamp", "available_time", *MODEL_COLUMNS}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError("Kronos history missing columns: " + ",".join(sorted(missing)))
    cutoff = pd.Timestamp(decision_time)
    if cutoff.tzinfo is None:
        raise ValueError("decision_time must be timezone-aware")
    cutoff_utc = cutoff.tz_convert("UTC")

    original_count = len(frame)
    data = frame.copy()
    available = pd.to_datetime(data["available_time"], errors="coerce")
    if available.isna().any() or available.dt.tz is None:
        raise ValueError("available_time must be valid and timezone-aware")
    data["available_time"] = available.dt.tz_convert("UTC")
    data = data.loc[data["available_time"] <= cutoff_utc].copy()
    if data.empty:
        raise ValueError("no bars were available at decision_time")

    timestamps = _market_local_naive(data["timestamp"])
    decision_local = cutoff.tz_convert("Asia/Shanghai").tz_localize(None)
    data["timestamp"] = timestamps
    data = data.loc[data["timestamp"] <= decision_local].copy()
    eligible_count_before_lookback = len(data)
    if data.empty:
        raise ValueError("no non-future bars remain at decision_time")
    if data["timestamp"].duplicated().any():
        raise ValueError("duplicate bar timestamps")
    data = data.sort_values("timestamp").tail(lookback).reset_index(drop=True)

    for column in MODEL_COLUMNS:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    values = data[list(MODEL_COLUMNS)].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Kronos OHLCV/amount contains missing or non-finite values")
    if (data[list(PRICE_COLUMNS)] <= 0).any().any():
        raise ValueError("Kronos prices must be positive")
    if (data[["volume", "amount"]] < 0).any().any():
        raise ValueError("Kronos volume and amount cannot be negative")
    valid_ohlc = (
        (data["high"] >= data[["open", "close", "low"]].max(axis=1))
        & (data["low"] <= data[["open", "close", "high"]].min(axis=1))
    )
    if not valid_ohlc.all():
        raise ValueError("invalid OHLC relationship")

    if len(data) < min(32, lookback):
        raise ValueError(f"insufficient known bars: {len(data)}")
    x_timestamp = pd.Series(pd.DatetimeIndex(data["timestamp"]))
    audit = {
        "input_bar_count": int(len(data)),
        "input_first_bar_time": data["timestamp"].iloc[0].isoformat(),
        "input_last_bar_time": data["timestamp"].iloc[-1].isoformat(),
        "decision_time": cutoff.isoformat(),
        "future_or_late_rows_excluded": int(original_count - eligible_count_before_lookback),
        "older_rows_trimmed_for_lookback": int(eligible_count_before_lookback - len(data)),
        "availability_rule": "available_time<=decision_time",
    }
    return data[list(MODEL_COLUMNS)].copy(), x_timestamp, audit


def summarize_forecast(prediction: pd.DataFrame, *, last_close: float) -> dict:
    """Convert forecast path to descriptive returns; never fabricate probabilities."""
    required = set(MODEL_COLUMNS)
    if not required.issubset(prediction.columns):
        raise ValueError("Kronos prediction missing required output columns")
    values = prediction[list(MODEL_COLUMNS)].apply(pd.to_numeric, errors="coerce")
    if values.empty or not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError("Kronos prediction is empty or non-finite")
    if (values[list(PRICE_COLUMNS)] <= 0).any().any():
        raise ValueError("Kronos predicted prices must be positive")
    if (values[["volume", "amount"]] < 0).any().any():
        raise ValueError("Kronos predicted volume and amount cannot be negative")
    if (values["high"] < values[["open", "close", "low"]].max(axis=1)).any():
        raise ValueError("Kronos predicted high violates OHLC")
    if (values["low"] > values[["open", "close", "high"]].min(axis=1)).any():
        raise ValueError("Kronos predicted low violates OHLC")
    if not np.isfinite(last_close) or last_close <= 0:
        raise ValueError("last_close must be positive and finite")

    closes = pd.concat([pd.Series([float(last_close)]), values["close"].reset_index(drop=True)])
    returns = {}
    for horizon in (1, 3, 5):
        if len(values) >= horizon:
            returns[f"forecast_return_{horizon}"] = float(closes.iloc[horizon] / last_close - 1.0)
    path = closes.to_numpy(dtype=float)
    peaks = np.maximum.accumulate(path)
    drawdown = path / peaks - 1.0
    predicted_returns = values["close"].pct_change().dropna()
    return {
        **returns,
        "forecast_range_pct": float(
            (values["high"].max() - values["low"].min()) / last_close
        ),
        "forecast_path_volatility": (
            float(predicted_returns.std(ddof=0)) if len(predicted_returns) else None
        ),
        "forecast_max_drawdown": float(drawdown.min()),
        "forecast_direction_5": (
            "UP" if returns.get("forecast_return_5", 0.0) > 0
            else "DOWN" if returns.get("forecast_return_5", 0.0) < 0
            else "FLAT"
        ) if "forecast_return_5" in returns else None,
        "probability": None,
        "trading_score": None,
    }


class KronosAdapter:
    """Lazy-loaded Kronos predictor; injected predictors keep tests offline."""

    def __init__(self, predictor: Any = None, config: KronosConfig | None = None):
        self.predictor = predictor
        self.config = config or KronosConfig()

    def _get_predictor(self):
        if self.predictor is None:
            if self.config.repo_path:
                upstream_root = str(Path(self.config.repo_path).expanduser().resolve())
                if upstream_root not in sys.path:
                    sys.path.insert(0, upstream_root)
            try:
                from model import Kronos, KronosPredictor, KronosTokenizer
            except ImportError as exc:
                raise RuntimeError(
                    "Kronos is optional and is not installed; install its pinned "
                    "upstream dependencies in a dedicated inference environment"
                ) from exc
            tokenizer = KronosTokenizer.from_pretrained(self.config.tokenizer_name)
            model = Kronos.from_pretrained(self.config.model_name)
            self.predictor = KronosPredictor(
                model, tokenizer, device=self.config.device,
                max_context=self.config.lookback,
            )
        return self.predictor

    def forecast(
        self,
        history: pd.DataFrame,
        *,
        decision_time: Any,
        future_timestamps: Sequence[Any],
    ) -> dict:
        """Forecast caller-supplied future sessions; calendar is never guessed here."""
        future = _market_local_naive(future_timestamps)
        if len(future) < 1 or len(future) > 20:
            raise ValueError("future_timestamps must contain between 1 and 20 sessions")
        if future.has_duplicates or not future.is_monotonic_increasing:
            raise ValueError("future_timestamps must be unique and increasing")

        model_frame, x_timestamp, audit = prepare_kronos_history(
            history, decision_time=decision_time, lookback=self.config.lookback,
        )
        last_bar = x_timestamp.iloc[-1]
        if (future <= last_bar).any():
            raise ValueError("future timestamps must be later than the last input bar")
        cutoff = pd.Timestamp(decision_time).tz_convert("Asia/Shanghai").tz_localize(None)
        if (future.normalize() <= cutoff.normalize()).any():
            raise ValueError("future timestamps must be after the decision session")

        predictor = self._get_predictor()
        prediction = predictor.predict(
            df=model_frame,
            x_timestamp=x_timestamp,
            y_timestamp=pd.Series(future),
            pred_len=len(future),
            T=self.config.temperature,
            top_p=self.config.top_p,
            sample_count=self.config.sample_count,
            verbose=False,
        )
        if not isinstance(prediction, pd.DataFrame) or len(prediction) != len(future):
            raise ValueError("Kronos returned an invalid prediction frame")
        prediction = prediction.copy()
        prediction.index = future
        metrics = summarize_forecast(
            prediction, last_close=float(model_frame["close"].iloc[-1]),
        )
        return {
            "status": "READY",
            "model_name": self.config.model_name,
            "tokenizer_name": self.config.tokenizer_name,
            "device": self.config.device,
            "generated_at": pd.Timestamp(decision_time).isoformat(),
            "historical_pit_eligible": False,
            "forecast_is_not_a_trading_signal": True,
            **audit,
            "metrics": metrics,
            "prediction": prediction,
        }
