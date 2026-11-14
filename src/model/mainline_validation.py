"""Outcome-only retrospective validation of detected mainline phase transitions.

Forward returns are labels for evaluation only. They must never be passed into
classify_mainlines or any live candidate/scoring path. All outputs remain
non-PIT because the source industry history lacks original availability times.
"""
from __future__ import annotations

import pandas as pd


REQUIRED = {
    "trade_date", "industry_code", "mainline_phase", "phase_transition",
    "industry_close", "market_close",
}


def evaluate_mainline_transitions(frame: pd.DataFrame, *, horizons=(5, 10, 20)) -> tuple[pd.DataFrame, dict]:
    missing = REQUIRED - set(frame.columns)
    if missing:
        raise ValueError("missing validation fields: " + ",".join(sorted(missing)))
    if not horizons or any(not isinstance(h, int) or h < 1 for h in horizons):
        raise ValueError("horizons must contain positive integer trading-session counts")
    out = frame.copy()
    out["trade_date"] = pd.to_datetime(out.trade_date, errors="coerce")
    for col in ("industry_close", "market_close"):
        out[col] = pd.to_numeric(out[col], errors="coerce")
    if out[list(REQUIRED)].isna().any().any():
        raise ValueError("validation inputs contain missing required observations")
    if out.duplicated(["industry_code", "trade_date"]).any():
        raise ValueError("duplicate industry/date rows in validation input")
    if (out[["industry_close", "market_close"]] <= 0).any().any():
        raise ValueError("validation closes must be positive")
    out = out.sort_values(["industry_code", "trade_date"]).reset_index(drop=True)
    for horizon in horizons:
        out[f"industry_forward_return_{horizon}d"] = (
            out.groupby("industry_code").industry_close.shift(-horizon)
            .div(out.industry_close).sub(1)
        )
        out[f"market_forward_return_{horizon}d"] = (
            out.groupby("industry_code").market_close.shift(-horizon)
            .div(out.market_close).sub(1)
        )
        out[f"excess_forward_return_{horizon}d"] = (
            out[f"industry_forward_return_{horizon}d"]
            - out[f"market_forward_return_{horizon}d"]
        )
    events = out[out.phase_transition & out.mainline_phase.isin(
        ["IGNITION", "PEAK", "DECAY"]
    )].copy()
    events["historical_pit_eligible"] = False
    summary = {}
    for phase in ("IGNITION", "PEAK", "DECAY"):
        phase_rows = events[events.mainline_phase.eq(phase)]
        phase_metrics = {"detected_transitions": int(len(phase_rows))}
        for horizon in horizons:
            values = phase_rows[f"excess_forward_return_{horizon}d"].dropna()
            phase_metrics[f"evaluated_{horizon}d"] = int(len(values))
            phase_metrics[f"mean_excess_{horizon}d"] = (
                float(values.mean()) if len(values) else None
            )
            favorable = values.gt(0) if phase == "IGNITION" else values.lt(0)
            phase_metrics[f"directional_hit_rate_{horizon}d"] = (
                float(favorable.mean()) if len(values) else None
            )
        summary[phase] = phase_metrics
    summary["interpretation"] = (
        "forward-outcome validation only; not causal, not point-in-time, "
        "and not a substitute for manually adjudicated historical episode labels"
    )
    summary["historical_pit_eligible"] = False
    return events.reset_index(drop=True), summary
