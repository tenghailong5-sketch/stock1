"""M5 adapter: evaluate complete, point-in-time factor rows through A/B/C rules.

This module is intentionally an adapter, not a data provider. Callers must
supply contemporaneous percentile factors and independently verified hard
gates. Missing inputs fail closed; no imputation or fabricated factor mapping
is performed here.
"""
from __future__ import annotations

from typing import Any, Mapping

import pandas as pd

from .r21_core import A_WEIGHTS, B_WEIGHTS, C_WEIGHTS, MarketState
from .scoring import (
    StrategyResult,
    candidate_band,
    qualify_candidate,
    score_a,
    score_b,
    score_c,
)


_GATE_KEYS = {
    "A": (
        "industry_top_30pct",
        "stock_residual_top_20pct",
        "amount_ma20_ge_100m",
        "breakout_or_confirmed_retest",
        "no_high_volume_stall_or_repeated_rejection",
        "next_open_gap_pct",
    ),
    "B": (
        "event_subtype",
        "event_data_verified",
        "liquidity_and_risk_pass",
        "announcement_risk_pass",
    ),
    "C": (
        "pre_pullback_residual_positive",
        "pullback_between_0_5_and_4pct",
        "pullback_volume_le_80pct_prior5d",
        "stop_or_reversal_confirmed",
        "industry_not_weak",
        "no_material_negative_event",
        "liquidity_and_risk_pass",
    ),
}


def _mapping(row: Mapping[str, Any], keys) -> dict[str, Any]:
    return {key: row[key] for key in keys if key in row}


def _invalid_state_result(strategy: str, stage: str, reason: str) -> StrategyResult:
    return StrategyResult(strategy, None, False, (reason,), stage)


def evaluate_row(row: Mapping[str, Any], *, stage: str = "close") -> list[dict[str, Any]]:
    """Evaluate one symbol snapshot; returns exactly one result per strategy."""
    symbol = str(row.get("symbol", "")).strip()
    if not symbol:
        raise ValueError("symbol is required")
    state_value = row.get("market_state")
    try:
        state = state_value if isinstance(state_value, MarketState) else MarketState(str(state_value))
    except (TypeError, ValueError):
        return [
            _result_dict(symbol, strategy, _invalid_state_result(
                strategy, stage, "missing_or_invalid_market_state"
            ), None)
            for strategy in ("A", "B", "C")
        ]

    definitions = (
        ("A", A_WEIGHTS, score_a, "close"),
        ("B", B_WEIGHTS, score_b, stage),
        ("C", C_WEIGHTS, score_c, "close"),
    )
    output = []
    for strategy, weights, scorer, default_stage in definitions:
        factors = _mapping(row, weights.keys())
        gates = _mapping(row, _GATE_KEYS[strategy])
        if strategy == "A":
            result = scorer(factors, gates, market_state=state, stage=default_stage)
        elif strategy == "B":
            result = scorer(factors, gates, market_state=state, stage=stage)
        else:
            result = scorer(factors, gates, market_state=state, stage=default_stage)

        # R1 M3 cooldown raises the floor in neutral markets. RISK_OFF is
        # independently rejected by the strategy scorers.
        floor = 85.0 if state == MarketState.NEUTRAL else 80.0
        result = qualify_candidate(result, minimum_score=floor)
        output.append(_result_dict(symbol, strategy, result, floor))
    return output


def _result_dict(symbol: str, strategy: str, result: StrategyResult,
                 minimum_score: float | None) -> dict[str, Any]:
    return {
        "symbol": symbol,
        "strategy": strategy,
        "stage": result.stage,
        "score": result.score,
        "eligible": result.eligible,
        "candidate_band": candidate_band(result.score) if result.score is not None else None,
        "minimum_score": minimum_score,
        "reasons": list(result.reasons),
    }


def evaluate_candidates(frame: pd.DataFrame, *, stage: str = "close") -> pd.DataFrame:
    """Evaluate rows without dropping rejected symbols or missing-data reasons."""
    if "symbol" not in frame.columns or "market_state" not in frame.columns:
        raise ValueError("candidate frame requires symbol and market_state")
    records = []
    for row in frame.to_dict(orient="records"):
        records.extend(evaluate_row(row, stage=stage))
    return pd.DataFrame.from_records(records, columns=[
        "symbol", "strategy", "stage", "score", "eligible",
        "candidate_band", "minimum_score", "reasons",
    ])
