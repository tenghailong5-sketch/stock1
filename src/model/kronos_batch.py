"""Bounded batch inference for the existing FHM candidate pool.

The caller owns candidate selection, exchange calendar and decision timestamp.
This module never scans the whole market or authorizes a trade.
"""
from __future__ import annotations

from typing import Mapping, Sequence, Any

import pandas as pd

from src.model.kronos_adapter import KronosAdapter, KronosConfig


HISTORY_COLUMN_MAP = {
    "date": "timestamp",
    "trade_date": "timestamp",
    "available_time": "available_time",
    "open": "open",
    "high": "high",
    "low": "low",
    "close": "close",
    "volume": "volume",
    "amount": "amount",
}


def normalize_candidate_history(frame: pd.DataFrame) -> pd.DataFrame:
    """Map fhm100.history() output to the strict Kronos adapter schema."""
    data = frame.rename(columns={
        key: value for key, value in HISTORY_COLUMN_MAP.items()
        if key in frame.columns and key != value
    }).copy()
    required = set(HISTORY_COLUMN_MAP.values())
    missing = required - set(data.columns)
    if missing:
        raise ValueError("candidate history missing: " + ",".join(sorted(missing)))
    return data[list(required)].copy()


def forecast_candidate_pool(
    histories: Mapping[str, pd.DataFrame],
    *,
    decision_time: Any,
    future_timestamps: Sequence[Any],
    adapter: KronosAdapter | None = None,
    config: KronosConfig | None = None,
    max_candidates: int = 60,
) -> dict:
    """Run one shared lazy-loaded predictor over a bounded, preselected pool."""
    if max_candidates < 1:
        raise ValueError("max_candidates must be positive")
    if len(histories) > max_candidates:
        raise ValueError(
            f"candidate pool {len(histories)} exceeds configured cap {max_candidates}"
        )
    if not histories:
        return {
            "status": "EMPTY",
            "decision_time": pd.Timestamp(decision_time).isoformat(),
            "candidate_count": 0,
            "results": {},
            "errors": {},
        }

    engine = adapter or KronosAdapter(config=config)
    results: dict[str, dict] = {}
    errors: dict[str, str] = {}
    for symbol, raw_history in histories.items():
        code = str(symbol).strip()
        if not code:
            errors[str(symbol)] = "empty symbol"
            continue
        try:
            normalized = normalize_candidate_history(raw_history)
            forecast = engine.forecast(
                normalized,
                decision_time=decision_time,
                future_timestamps=future_timestamps,
            )
            forecast["symbol"] = code
            forecast["historical_pit_eligible"] = False
            results[code] = forecast
        except Exception as exc:
            errors[code] = f"{type(exc).__name__}: {str(exc)[:240]}"

    return {
        "status": "READY" if results else "NO_VALID_FORECASTS",
        "decision_time": pd.Timestamp(decision_time).isoformat(),
        "candidate_count": len(histories),
        "forecast_count": len(results),
        "error_count": len(errors),
        "model_name": engine.config.model_name,
        "historical_pit_eligible": False,
        "forecasts_are_not_trading_signals": True,
        "results": results,
        "errors": errors,
    }
