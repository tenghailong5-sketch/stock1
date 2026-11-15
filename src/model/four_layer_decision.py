"""FHM-100 four-layer decision gate.

This module links M2 market permission, M3 industry lifecycle, M5 candidate
eligibility and M8 pre-trade risk. It is intentionally fail-closed: diagnostic
or non-PIT industry observations cannot authorize a formal entry.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any


@dataclass(frozen=True)
class MarketPermission:
    mode: str
    risk_scale: float
    allow_new_entries: bool
    reason: str


@dataclass(frozen=True)
class IndustryPermission:
    mode: str
    allow_new_entries: bool
    reason: str


@dataclass(frozen=True)
class TradeDecision:
    allowed: bool
    market_mode: str
    industry_mode: str
    max_allowed_risk: float
    planned_risk: float
    reasons: tuple[str, ...]


def _finite(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return False
    return math.isfinite(number)


def market_permission(market_state: str, position_scale: float | None,
                      *, data_ready: bool) -> MarketPermission:
    """Translate M2's confirmed state into entry permission and risk scale."""
    if data_ready is not True or not _finite(position_scale):
        return MarketPermission("BLOCKED", 0.0, False, "market_context_not_ready")
    scale = float(position_scale)
    if not 0.0 <= scale <= 1.0:
        return MarketPermission("BLOCKED", 0.0, False, "invalid_market_position_scale")
    if market_state == "RISK_ON" and scale > 0:
        return MarketPermission("ATTACK", scale, True, "confirmed_risk_on")
    if market_state == "NEUTRAL" and scale > 0:
        return MarketPermission("PROBE", scale, True, "confirmed_neutral_dynamic_scale")
    if market_state == "RISK_OFF":
        return MarketPermission("DEFEND", 0.0, False, "confirmed_risk_off")
    return MarketPermission("BLOCKED", 0.0, False, "unknown_market_state_or_zero_risk_scale")


def industry_permission(
    mainline_phase: str,
    *,
    persistence_ready: bool,
    historical_pit_eligible: bool,
    stock_turnover_pit_status: str,
    decision_mode: str = "historical_pit",
    industry_live_data_eligible: bool = False,
    stock_turnover_live_status: str = "not_ready",
) -> IndustryPermission:
    """Only a PIT-verified, ready ignition phase may authorize a fresh entry.

    PEAK is hold/manage-only; DECAY is blocked. TRANSITION/UNKNOWN and all
    diagnostic or incomplete inputs are blocked rather than treated as neutral.
    """
    if decision_mode == "historical_pit":
        if persistence_ready is not True:
            return IndustryPermission("BLOCKED", False, "industry_persistence_not_ready")
        if historical_pit_eligible is not True:
            return IndustryPermission("BLOCKED", False, "industry_history_not_pit_eligible")
        if stock_turnover_pit_status != "ready":
            return IndustryPermission("BLOCKED", False, "stock_turnover_concentration_not_ready")
    elif decision_mode == "live_forward":
        if industry_live_data_eligible is not True:
            return IndustryPermission("BLOCKED", False, "industry_live_snapshot_not_ready")
        if persistence_ready is not True:
            return IndustryPermission("BLOCKED", False, "industry_persistence_not_ready")
        # Stock-level turnover concentration remains a ranking/diagnostic input
        # in live mode, not a second hard gate. The live industry snapshot and
        # the M3 persistence test are both mandatory before an ignition entry.
    else:
        return IndustryPermission("BLOCKED", False, "invalid_decision_mode")
    if mainline_phase == "IGNITION":
        reason = (
            "live_forward_mainline_ignition"
            if decision_mode == "live_forward"
            else "pit_verified_mainline_ignition"
        )
        return IndustryPermission("ENTRY_ALLOWED", True, reason)
    if mainline_phase == "PEAK":
        return IndustryPermission("HOLD_ONLY", False, "mainline_peak_no_fresh_entry")
    if mainline_phase == "DECAY":
        return IndustryPermission("EXIT_OR_DEFEND", False, "mainline_decay")
    return IndustryPermission("BLOCKED", False, "mainline_phase_unknown_or_transition")


def evaluate_trade_decision(
    *,
    market_state: str,
    market_position_scale: float | None,
    market_data_ready: bool,
    mainline_phase: str,
    industry_persistence_ready: bool,
    industry_historical_pit_eligible: bool,
    stock_turnover_pit_status: str,
    decision_mode: str = "historical_pit",
    industry_live_data_eligible: bool = False,
    stock_turnover_live_status: str = "not_ready",
    strategy_eligible: bool,
    strategy_score: float | None,
    entry_ready: bool,
    entry_price: float,
    stop_price: float,
    quantity: int,
    planned_risk: float,
    reward_risk: float,
    minimum_reward_risk: float,
    existing_portfolio_risk: float,
    existing_position_count: int,
    available_risk_budget: float,
    same_industry_or_corr_gt_06: bool = False,
) -> TradeDecision:
    """Apply M2 -> M3 -> M5 -> M8 gates before a new long position.

    planned_risk must include estimated fees, slippage and gap allowance.
    The gross stop-distance risk is independently recomputed as a lower bound.
    No fixed score, position-count, per-name, or portfolio-currency caps are
    imposed here. M8 consumes the strategy/risk engine's dynamic available
    risk budget, already net of current exposure and portfolio constraints.
    """
    market = market_permission(
        market_state, market_position_scale, data_ready=market_data_ready
    )
    industry = industry_permission(
        mainline_phase,
        persistence_ready=industry_persistence_ready,
        historical_pit_eligible=industry_historical_pit_eligible,
        stock_turnover_pit_status=stock_turnover_pit_status,
        decision_mode=decision_mode,
        industry_live_data_eligible=industry_live_data_eligible,
        stock_turnover_live_status=stock_turnover_live_status,
    )
    reasons: list[str] = []
    if not market.allow_new_entries:
        reasons.append(market.reason)
    if not industry.allow_new_entries:
        reasons.append(industry.reason)
    if strategy_eligible is not True:
        reasons.append("M5_strategy_not_eligible")
    if not _finite(strategy_score) or not 0.0 <= float(strategy_score) <= 100.0:
        reasons.append("M5_score_missing_or_out_of_range")
    if entry_ready is not True:
        reasons.append("entry_not_confirmed")
    if not isinstance(same_industry_or_corr_gt_06, bool):
        reasons.append("invalid_industry_correlation_flag")
    elif same_industry_or_corr_gt_06:
        reasons.append("same_industry_or_high_correlation_exposure")
    if isinstance(existing_position_count, bool) or not isinstance(existing_position_count, int) or existing_position_count < 0:
        reasons.append("invalid_existing_position_count")
    if not _finite(entry_price) or float(entry_price) <= 0:
        reasons.append("invalid_entry_price")
    if not _finite(stop_price) or float(stop_price) <= 0:
        reasons.append("invalid_stop_price")
    if _finite(entry_price) and _finite(stop_price) and float(stop_price) >= float(entry_price):
        reasons.append("long_stop_must_be_below_entry")
    if isinstance(quantity, bool) or not isinstance(quantity, int) or quantity <= 0:
        reasons.append("invalid_quantity")
    if not _finite(planned_risk) or float(planned_risk) <= 0:
        reasons.append("invalid_planned_risk")
    if not _finite(reward_risk) or not _finite(minimum_reward_risk) or float(minimum_reward_risk) <= 0:
        reasons.append("invalid_reward_risk_inputs")
    elif float(reward_risk) < float(minimum_reward_risk):
        reasons.append("reward_risk_below_configured_minimum")
    if not _finite(existing_portfolio_risk) or float(existing_portfolio_risk) < 0:
        reasons.append("invalid_existing_portfolio_risk")
    if not _finite(available_risk_budget) or float(available_risk_budget) <= 0:
        reasons.append("dynamic_available_risk_budget_missing_or_invalid")

    if _finite(entry_price) and _finite(stop_price) and isinstance(quantity, int) and not isinstance(quantity, bool):
        gross_stop_risk = max(0.0, (float(entry_price) - float(stop_price)) * quantity)
        if _finite(planned_risk) and float(planned_risk) + 1e-9 < gross_stop_risk:
            reasons.append("planned_risk_understates_gross_stop_risk")
    else:
        gross_stop_risk = float("nan")

    max_allowed_risk = (
        float(available_risk_budget) * market.risk_scale
        if market.allow_new_entries and _finite(available_risk_budget)
        and float(available_risk_budget) > 0 else 0.0
    )

    if _finite(planned_risk) and float(planned_risk) > max_allowed_risk:
        reasons.append("planned_risk_exceeds_dynamic_available_budget")

    return TradeDecision(
        allowed=not reasons,
        market_mode=market.mode,
        industry_mode=industry.mode,
        max_allowed_risk=float(max_allowed_risk),
        planned_risk=float(planned_risk) if _finite(planned_risk) else 0.0,
        reasons=tuple(reasons),
    )
