"""FHM-100 R2.1 safety primitives.

These are research controls, not a complete trading engine. Unknown data or
execution states fail closed. All timestamps must be timezone-aware.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Iterable, Mapping


A_WEIGHTS = {
    "industry_relative_strength": 20,
    "stock_residual_strength": 25,
    "price_structure": 15,
    "trade_quality": 15,
    "market_environment": 15,
    "catalyst": 10,
}
B_WEIGHTS = {
    "event_type": 20,
    "pre_event_trend": 15,
    "event_trade_structure": 20,
    "industry_market_environment": 15,
    "next_day_price_action": 20,
    "risk": 10,
}
C_WEIGHTS = {
    "industry_strength": 15,
    "pre_pullback_strength": 15,
    "pullback_reversal_structure": 25,
    "trade_quality": 20,
    "market_environment": 15,
    "catalyst_risk": 10,
}


class MarketState(str, Enum):
    RISK_ON = "RISK_ON"
    NEUTRAL = "NEUTRAL"
    RISK_OFF = "RISK_OFF"


class ExitState(str, Enum):
    OPEN = "OPEN"
    EXIT_TRIGGERED = "EXIT_TRIGGERED"
    EXIT_PENDING = "EXIT_PENDING"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    CLOSED = "CLOSED"
    SUSPENDED_PENDING = "SUSPENDED_PENDING"
    LIMIT_DOWN_LOCKED = "LIMIT_DOWN_LOCKED"
    NO_LIQUIDITY = "NO_LIQUIDITY"


def point_in_time_rows(rows: Iterable[Mapping], decision_time: datetime) -> list[Mapping]:
    """Only return rows known to the system at decision_time; unknown times fail closed."""
    if decision_time.tzinfo is None:
        raise ValueError("decision_time must be timezone-aware")
    out = []
    for row in rows:
        available = row.get("available_time")
        if not isinstance(available, datetime) or available.tzinfo is None:
            continue
        if available <= decision_time:
            out.append(row)
    return out


def validate_weights(weights: Mapping[str, float]) -> None:
    if not weights or any(v < 0 for v in weights.values()):
        raise ValueError("weights must be non-empty and non-negative")
    if abs(sum(weights.values()) - 100.0) > 1e-9:
        raise ValueError("strategy weights must sum to 100")


def weighted_score(weights: Mapping[str, float], factors: Mapping[str, float]) -> float:
    """No silent imputation: every weighted factor must be present and in [0,100]."""
    validate_weights(weights)
    missing = set(weights) - set(factors)
    if missing:
        raise ValueError(f"missing factors: {','.join(sorted(missing))}")
    values = [factors[k] for k in weights]
    if any(isinstance(v, bool) or v is None or not 0 <= float(v) <= 100
           for v in values):
        raise ValueError("factor values must be finite numbers in [0,100]")
    return sum(weights[k] * float(factors[k]) for k in weights) / 100.0


# Market-state thresholds are deliberately not hard-coded: the model requires
# point-in-time calibration on historical data before M3 can classify states.

def position_risk_budget(q_rank: int, first_used: float = 0.0,
                         same_industry_or_corr_gt_06: bool = False) -> float:
    """R1 S2+S3 allocation, preserving 2,000 per-name and 3,000 portfolio caps."""
    if q_rank not in (1, 2):
        return 0.0
    remaining = max(0.0, 3000.0 - first_used)
    budget = min(2000.0, remaining) if q_rank == 1 else min(1000.0, remaining)
    if q_rank == 2 and same_industry_or_corr_gt_06:
        budget *= 0.5
    return budget if budget >= 600.0 else 0.0


def can_open_position(*, a_data_valid: bool, ruleset_known: bool,
                      corporate_actions_resolved: bool, risk_inputs_valid: bool,
                      market: MarketState, prediction_available: bool) -> bool:
    """Fail closed on missing A-grade data, rules, corporate actions or risk inputs."""
    return all((a_data_valid, ruleset_known, corporate_actions_resolved,
                risk_inputs_valid, prediction_available)) and market != MarketState.RISK_OFF


def conservative_daily_exit(open_price: float, high: float, low: float,
                            take_profit: float, stop_loss: float) -> str:
    """OHLC cannot reveal intraday order; if both barriers touched, assume stop first."""
    if open_price <= stop_loss:
        return "STOP_AT_OPEN"
    hit_stop = low <= stop_loss
    hit_take = high >= take_profit
    if hit_stop:
        return "STOP_FIRST" if hit_take else "STOP"
    if hit_take:
        return "TAKE_PROFIT"
    return "HOLD"


@dataclass
class PositionExit:
    state: ExitState = ExitState.OPEN
    remaining_qty: int = 0

    def trigger(self) -> None:
        if self.state == ExitState.OPEN:
            self.state = ExitState.EXIT_TRIGGERED
            self.state = ExitState.EXIT_PENDING

    def apply_fill(self, qty: int) -> None:
        if qty < 0 or qty > self.remaining_qty:
            raise ValueError("fill quantity exceeds remaining position")
        if self.state not in (ExitState.EXIT_PENDING, ExitState.PARTIALLY_FILLED):
            raise ValueError("position is not awaiting exit")
        self.remaining_qty -= qty
        self.state = ExitState.CLOSED if self.remaining_qty == 0 else ExitState.PARTIALLY_FILLED

    def blocked(self, reason: ExitState) -> None:
        if reason not in (ExitState.SUSPENDED_PENDING, ExitState.LIMIT_DOWN_LOCKED,
                          ExitState.NO_LIQUIDITY):
            raise ValueError("invalid blocked state")
        if self.state != ExitState.CLOSED:
            self.state = reason
