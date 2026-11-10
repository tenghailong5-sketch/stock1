"""FHM-100 M5 independent strategy scoring and hard gates.

Scores are only valid when every weighted percentile is observed. Hard gates are
evaluated separately and can never be offset by a high score. This module does
not invent factor values or infer missing event/turnover/news data.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Any

from .r21_core import A_WEIGHTS, B_WEIGHTS, C_WEIGHTS, MarketState, weighted_score


@dataclass(frozen=True)
class StrategyResult:
    strategy: str
    score: float | None
    eligible: bool
    reasons: tuple[str, ...]
    stage: str


def _finite_number(value: Any) -> bool:
    try:
        return value is not None and float(value) == float(value) and abs(float(value)) != float("inf")
    except (TypeError, ValueError):
        return False


def _require_bool(data: Mapping[str, Any], key: str, reasons: list[str]) -> bool:
    value = data.get(key)
    if not isinstance(value, bool):
        reasons.append(f"missing_or_invalid:{key}")
        return False
    if value is False:
        reasons.append(f"hard_gate_failed:{key}")
        return False
    return True


def _score(strategy: str, weights: Mapping[str, int],
           factors: Mapping[str, Any], *, stage: str) -> StrategyResult:
    reasons: list[str] = []
    if strategy == "B" and stage != "next_day":
        if "next_day_price_action" in factors:
            reasons.append("lookahead_forbidden:next_day_price_action")
        return StrategyResult(strategy, None, False, tuple(reasons or ["B_requires_next_day_stage"]), stage)
    missing = [k for k in weights if k not in factors or not _finite_number(factors.get(k))]
    if missing:
        reasons.extend(f"missing_or_invalid_factor:{k}" for k in missing)
        return StrategyResult(strategy, None, False, tuple(reasons), stage)
    try:
        score = weighted_score(weights, factors)
    except (TypeError, ValueError) as exc:
        return StrategyResult(strategy, None, False, (f"score_error:{exc}",), stage)
    return StrategyResult(strategy, score, True, (), stage)


def score_a(factors: Mapping[str, Any], gates: Mapping[str, Any],
            *, market_state: MarketState, stage: str = "close") -> StrategyResult:
    """A: trend continuation; threshold rules are independent of weighted score."""
    scored = _score("A", A_WEIGHTS, factors, stage=stage)
    if not scored.eligible:
        return scored
    reasons: list[str] = []
    if market_state == MarketState.RISK_OFF:
        reasons.append("market_risk_off")
    for key in ("industry_top_30pct", "stock_residual_top_20pct",
                "amount_ma20_ge_100m", "breakout_or_confirmed_retest",
                "no_high_volume_stall_or_repeated_rejection"):
        _require_bool(gates, key, reasons)
    if gates.get("next_open_gap_pct") is not None:
        if not _finite_number(gates["next_open_gap_pct"]):
            reasons.append("invalid:next_open_gap_pct")
        elif float(gates["next_open_gap_pct"]) > 3.0:
            reasons.append("hard_gate_failed:next_open_gap_gt_3pct")
    return StrategyResult("A", scored.score, not reasons, tuple(reasons), stage)


def score_b(factors: Mapping[str, Any], gates: Mapping[str, Any],
            *, market_state: MarketState, stage: str) -> StrategyResult:
    """B: event strategy; next-day component is only scored after it exists."""
    if stage != "next_day":
        return StrategyResult("B", None, False, ("B_requires_next_day_stage",), stage)
    event_type = gates.get("event_subtype")
    supported_events = {"hit_up_unsealed", "close_up_sealed", "hit_down_unsealed", "close_down_sealed"}
    if event_type not in supported_events:
        return StrategyResult("B", None, False, ("missing_or_unsupported_event_subtype",), stage)
    scored = _score("B", B_WEIGHTS, factors, stage=stage)
    if not scored.eligible:
        return scored
    reasons: list[str] = []
    if market_state == MarketState.RISK_OFF:
        reasons.append("market_risk_off")
    for key in ("event_data_verified", "liquidity_and_risk_pass",
                "announcement_risk_pass"):
        _require_bool(gates, key, reasons)
    return StrategyResult("B", scored.score, not reasons, tuple(reasons), stage)


def score_c(factors: Mapping[str, Any], gates: Mapping[str, Any],
            *, market_state: MarketState, stage: str = "close") -> StrategyResult:
    """C: strong-stock pullback reversal with explicit structural gates."""
    scored = _score("C", C_WEIGHTS, factors, stage=stage)
    if not scored.eligible:
        return scored
    reasons: list[str] = []
    if market_state == MarketState.RISK_OFF:
        reasons.append("market_risk_off")
    for key in ("pre_pullback_residual_positive", "pullback_between_0_5_and_4pct",
                "pullback_volume_le_80pct_prior5d", "stop_or_reversal_confirmed",
                "industry_not_weak", "no_material_negative_event",
                "liquidity_and_risk_pass"):
        _require_bool(gates, key, reasons)
    return StrategyResult("C", scored.score, not reasons, tuple(reasons), stage)


def candidate_band(score: float) -> str:
    """Descriptive research band only; never an order instruction."""
    if not _finite_number(score) or not 0 <= float(score) <= 100:
        raise ValueError("score must be finite and within [0,100]")
    if score >= 90:
        return "priority_research"
    if score >= 80:
        return "candidate"
    if score >= 70:
        return "watch"
    return "below_threshold"


def qualify_candidate(result: StrategyResult, *, minimum_score: float = 80.0) -> StrategyResult:
    """Apply the non-negotiable FHM candidate floor after all strategy hard gates.

    The configured floor may be raised but never lowered below 80. A high score
    cannot override failed/missing hard gates, and a missing score never qualifies.
    """
    if not _finite_number(minimum_score) or not 80.0 <= float(minimum_score) <= 100.0:
        raise ValueError("minimum_score must be finite and within [80,100]")
    if not result.eligible or result.score is None:
        return result
    if not _finite_number(result.score) or not 0.0 <= float(result.score) <= 100.0:
        return StrategyResult(result.strategy, None, False,
                              result.reasons + ("invalid_score",), result.stage)
    if float(result.score) < float(minimum_score):
        return StrategyResult(result.strategy, result.score, False,
                              result.reasons + (f"below_minimum_score:{minimum_score:g}",),
                              result.stage)
    return result


def score_factor_bundle(frame, factor_map, *, date_col="trade_date"):
    """Rank raw factors within each trading date; missing values stay missing."""
    from .m4_factors import rank_factor_by_date
    out = frame.copy()
    for output_name, spec in factor_map.items():
        if not isinstance(spec, tuple) or len(spec) != 2:
            raise ValueError("invalid factor mapping for " + output_name)
        raw_col, higher_is_better = spec
        if raw_col not in out:
            raise ValueError("missing raw factor column: " + raw_col)
        out[output_name] = rank_factor_by_date(out, raw_col, date_col=date_col,
                                               higher_is_better=bool(higher_is_better))
    return out
