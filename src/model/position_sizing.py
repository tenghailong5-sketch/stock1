"""Fail-closed A-share long position sizing from explicit portfolio risk limits.

Research sizing only; this module never places orders. Estimated loss includes
the supplied stop distance and estimated round-trip transaction costs.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class SizingResult:
    approved: bool
    shares: int
    entry_price: float
    stop_price: float
    estimated_loss: float
    risk_budget: float
    cash_required: float
    reasons: tuple[str, ...]

    def to_dict(self) -> dict:
        result = asdict(self)
        result["reasons"] = list(self.reasons)
        return result


def size_long_position(
    *, entry_price: float, stop_price: float, available_cash: float,
    open_position_count: int, open_portfolio_risk: float,
    per_trade_risk_limit: float = 2000.0,
    portfolio_risk_limit: float = 3000.0,
    max_positions: int = 2, lot_size: int = 100,
    estimated_round_trip_cost_bps: float = 30.0,
    estimated_round_trip_fixed_cost: float = 10.0,
    estimated_buy_commission_bps: float = 2.5,
    estimated_buy_minimum_commission: float = 5.0,
    estimated_buy_transfer_fee_bps: float = 0.1,
) -> SizingResult:
    """Size the largest board-lot long position within cash and loss limits.

    open_portfolio_risk is the sum of remaining stop risks of existing
    positions, including estimated costs. Long positions require stop < entry.
    """
    values = {
        "entry_price": entry_price, "stop_price": stop_price,
        "available_cash": available_cash, "open_portfolio_risk": open_portfolio_risk,
        "per_trade_risk_limit": per_trade_risk_limit,
        "portfolio_risk_limit": portfolio_risk_limit,
        "estimated_round_trip_cost_bps": estimated_round_trip_cost_bps,
        "estimated_round_trip_fixed_cost": estimated_round_trip_fixed_cost,
        "estimated_buy_commission_bps": estimated_buy_commission_bps,
        "estimated_buy_minimum_commission": estimated_buy_minimum_commission,
        "estimated_buy_transfer_fee_bps": estimated_buy_transfer_fee_bps,
    }
    reasons = []
    for name, value in values.items():
        if isinstance(value, bool):
            reasons.append(f"invalid:{name}")
            continue
        try:
            if not math.isfinite(float(value)):
                reasons.append(f"invalid:{name}")
        except (TypeError, ValueError, OverflowError):
            reasons.append(f"invalid:{name}")
    if isinstance(open_position_count, bool) or not isinstance(open_position_count, int) or open_position_count < 0:
        reasons.append("invalid:open_position_count")
    if isinstance(max_positions, bool) or not isinstance(max_positions, int) or max_positions < 1:
        reasons.append("invalid:max_positions")
    if isinstance(lot_size, bool) or not isinstance(lot_size, int) or lot_size < 1:
        reasons.append("invalid:lot_size")
    if reasons:
        return SizingResult(False, 0, 0.0, 0.0, 0.0, 0.0, 0.0, tuple(reasons))

    entry, stop, cash = float(entry_price), float(stop_price), float(available_cash)
    existing_risk = float(open_portfolio_risk)
    trade_limit, portfolio_limit = float(per_trade_risk_limit), float(portfolio_risk_limit)
    cost_bps = float(estimated_round_trip_cost_bps)
    fixed_cost = float(estimated_round_trip_fixed_cost)
    buy_commission_bps = float(estimated_buy_commission_bps)
    buy_minimum_commission = float(estimated_buy_minimum_commission)
    buy_transfer_bps = float(estimated_buy_transfer_fee_bps)
    if entry <= 0 or stop <= 0 or stop >= entry:
        reasons.append("invalid_long_entry_or_stop")
    if cash < 0 or existing_risk < 0 or trade_limit <= 0 or portfolio_limit <= 0 or cost_bps < 0 or fixed_cost < 0 or buy_commission_bps < 0 or buy_minimum_commission < 0 or buy_transfer_bps < 0:
        reasons.append("negative_or_nonpositive_risk_input")
    if open_position_count >= max_positions:
        reasons.append("maximum_open_positions_reached")
    remaining = portfolio_limit - existing_risk
    if remaining <= 0:
        reasons.append("portfolio_risk_budget_exhausted")
    if reasons:
        return SizingResult(False, 0, entry, stop, 0.0, 0.0, 0.0, tuple(reasons))

    budget = min(trade_limit, remaining)
    cost_per_share = entry * cost_bps / 10000.0
    loss_per_share = (entry - stop) + cost_per_share
    risk_after_fixed = max(0.0, budget - fixed_cost)
    risk_shares = math.floor((risk_after_fixed / loss_per_share) / lot_size) * lot_size
    # Reserve buy-side fees before deciding how many board lots cash can fund.
    cash_shares = 0
    max_cash_lots = math.floor((cash / entry) / lot_size)
    low_lots, high_lots = 0, max_cash_lots
    while low_lots <= high_lots:
        mid_lots = (low_lots + high_lots) // 2
        qty = mid_lots * lot_size
        turnover = qty * entry
        buy_fee = (max(buy_minimum_commission, turnover * buy_commission_bps / 10000.0)
                   + turnover * buy_transfer_bps / 10000.0) if qty else 0.0
        if turnover + buy_fee <= cash + 1e-9:
            cash_shares = qty
            low_lots = mid_lots + 1
        else:
            high_lots = mid_lots - 1
    shares = max(0, min(risk_shares, cash_shares))
    loss = shares * loss_per_share + (fixed_cost if shares else 0.0)
    required_cash = shares * entry
    reasons.append("sized_within_trade_portfolio_and_cash_limits" if shares else
                   "no_affordable_board_lot_within_risk_budget")
    return SizingResult(
        shares > 0, shares, entry, stop, round(loss, 2), round(budget, 2),
        round(required_cash, 2), tuple(reasons)
    )
