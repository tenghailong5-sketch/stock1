"""Checkpointed, date-batched portfolio backtest runner.

Each checkpoint is written only after a complete trading session batch. The
underlying portfolio state is carried forward, so batches are not independent
simulations and preserve cash, positions, pending stops and audit trails.
"""
from __future__ import annotations

from dataclasses import asdict
from datetime import date
import hashlib
import json
from typing import Any

import pandas as pd

from .execution import ExecutionCosts
from .portfolio_backtest import _as_date, run_portfolio_backtest
from .run_checkpoint import SQLiteCheckpointStore


def _frame_fingerprint(frame: pd.DataFrame) -> str:
    normalized = frame.copy().astype(str)
    digest = hashlib.sha256()
    digest.update("|".join(map(str, normalized.columns)).encode("utf-8"))
    digest.update(pd.util.hash_pandas_object(normalized, index=True).values.tobytes())
    return digest.hexdigest()


def run_checkpointed_portfolio_backtest(
    candidates: pd.DataFrame,
    bars: pd.DataFrame,
    *,
    checkpoint_path: str,
    run_id: str,
    batch_size: int = 20,
    initial_cash: float = 100000.0,
    max_participation: float = 0.05,
    costs: ExecutionCosts = ExecutionCosts(),
) -> dict[str, Any]:
    """Run/resume a portfolio simulation using deterministic date batches."""
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    if not run_id or not str(run_id).strip():
        raise ValueError("run_id is required")

    # The core engine performs the complete PIT/data admission audit before
    # the first batch. Its date mapping is based on the full input history.
    if "trade_date" not in bars.columns:
        raise ValueError("bars_missing_trade_date")
    dates = sorted({_as_date(value) for value in bars["trade_date"]})
    if not dates:
        return run_portfolio_backtest(
            candidates, bars, initial_cash=initial_cash,
            max_participation=max_participation, costs=costs,
        )

    batches = [
        dates[index:index + batch_size]
        for index in range(0, len(dates), batch_size)
    ]
    data_fingerprint = hashlib.sha256(
        (_frame_fingerprint(candidates) + _frame_fingerprint(bars)).encode("ascii")
    ).hexdigest()
    config = {
        "initial_cash": float(initial_cash),
        "max_participation": float(max_participation),
        "costs": asdict(costs),
        "batch_size": batch_size,
        "first_date": dates[0].isoformat(),
        "last_date": dates[-1].isoformat(),
    }
    config_fingerprint = hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    run_fingerprint = hashlib.sha256(
        (data_fingerprint + config_fingerprint).encode("ascii")
    ).hexdigest()

    store = SQLiteCheckpointStore(checkpoint_path)
    stage = "portfolio_backtest"
    store.start_run(run_id, stage, run_fingerprint, {
        "data_fingerprint": data_fingerprint,
        "config": config,
        "date_count": len(dates),
    })
    batch_fingerprints = {}
    for index, batch_dates in enumerate(batches):
        batch_id = f"{index:06d}-{batch_dates[0].isoformat()}-{batch_dates[-1].isoformat()}"
        batch_fingerprints[batch_id] = hashlib.sha256(
            (run_fingerprint + batch_id).encode("ascii")
        ).hexdigest()
    store.register_batches(run_id, stage, batch_fingerprints)

    state = None
    latest_result = None
    for index, batch_dates in enumerate(batches):
        batch_id = f"{index:06d}-{batch_dates[0].isoformat()}-{batch_dates[-1].isoformat()}"
        summary = store.summary(run_id, stage)
        # Completed batches are immutable and are used to restore the exact
        # state at the end of each prior batch.
        if batch_id not in store.pending_batches(run_id, stage):
            saved = store.load_result(run_id, stage, batch_id)
            state = saved["state"]
            latest_result = saved["result"]
            continue

        store.mark_started(run_id, stage, batch_id)
        try:
            latest_result = run_portfolio_backtest(
                candidates, bars,
                initial_cash=initial_cash,
                max_participation=max_participation,
                costs=costs,
                _state=state,
                _date_start=batch_dates[0],
                _date_end=batch_dates[-1],
                _include_state=True,
            )
            state = latest_result.pop("_state")
            store.mark_completed(run_id, stage, batch_id, {
                "state": state,
                "result": latest_result,
            })
        except Exception as exc:
            store.mark_failed(run_id, stage, batch_id, f"{type(exc).__name__}: {exc}")
            raise

    if latest_result is None:
        raise RuntimeError("checkpoint run has no completed batch result")
    return {**latest_result, "checkpoint": {
        "run_id": run_id,
        "stage": stage,
        **store.summary(run_id, stage),
    }}
