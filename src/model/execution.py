"""Conservative daily-bar execution rules for A-share research backtests.

This is a deterministic simulator, not a broker adapter. It requires explicit
limit/suspension flags because daily OHLC alone cannot prove queue execution.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
from datetime import date
import math
from numbers import Real


@dataclass(frozen=True)
class ExecutionCosts:
    commission_bps: float = 2.5
    minimum_commission: float = 5.0
    sell_stamp_duty_bps: float = 5.0
    slippage_bps: float = 10.0
    transfer_fee_bps: float = 0.1

    def __post_init__(self):
        values = (self.commission_bps, self.minimum_commission,
                  self.sell_stamp_duty_bps, self.slippage_bps,
                  self.transfer_fee_bps)
        if any(isinstance(value, bool) or not isinstance(value, Real)
               or not math.isfinite(float(value)) or float(value) < 0
               for value in values):
            raise ValueError("execution costs must be finite non-negative numbers")


@dataclass(frozen=True)
class FillResult:
    filled: bool
    side: str
    quantity: int
    reference_price: float
    fill_price: float | None
    commission: float
    stamp_duty: float
    transfer_fee: float
    total_cost: float
    reason: str

    def to_dict(self) -> dict:
        return asdict(self)


def _finite_positive(value: float) -> bool:
    try:
        return (not isinstance(value, bool) and isinstance(value, Real)
                and math.isfinite(float(value)) and float(value) > 0)
    except (TypeError, ValueError, OverflowError):
        return False


def simulate_daily_fill(
    *, side: str, quantity: int, reference_price: float,
    trade_date: date, entry_date: date | None = None,
    suspended: bool = False, limit_up_locked: bool = False,
    limit_down_locked: bool = False, available_volume: int | None = None,
    max_participation: float = 0.05, lot_size: int = 100,
    costs: ExecutionCosts = ExecutionCosts(),
) -> FillResult:
    """Simulate a conservative board-lot fill; unobservable queues are rejected.

    Buy fills are blocked at a locked upper limit; sell fills at a locked lower
    limit. T+1 prevents selling on the entry date. Volume participation caps
    fills to a configured fraction of reported daily volume.
    """
    if side not in ("buy", "sell"):
        raise ValueError("side must be buy or sell")
    if isinstance(quantity, bool) or not isinstance(quantity, int) or quantity <= 0:
        raise ValueError("quantity must be a positive integer")
    if not _finite_positive(reference_price):
        raise ValueError("reference_price must be finite and positive")
    if not isinstance(trade_date, date) or (entry_date is not None and not isinstance(entry_date, date)):
        raise ValueError("trade_date and entry_date must be dates")
    if not isinstance(costs, ExecutionCosts):
        raise ValueError("costs must be an ExecutionCosts instance")
    for flag_name, flag_value in (("suspended", suspended),
                                  ("limit_up_locked", limit_up_locked),
                                  ("limit_down_locked", limit_down_locked)):
        if not isinstance(flag_value, bool):
            raise ValueError(flag_name + " must be an explicit boolean")
    if (isinstance(max_participation, bool) or not isinstance(max_participation, Real)
            or not math.isfinite(float(max_participation))
            or not 0 < float(max_participation) <= 1):
        raise ValueError("max_participation must be a finite number in (0, 1]")
    if isinstance(lot_size, bool) or not isinstance(lot_size, int) or lot_size <= 0:
        raise ValueError("lot_size must be a positive integer")
    if quantity % lot_size:
        raise ValueError("quantity must be a board-lot multiple")
    reason = None
    if suspended:
        reason = "suspended_no_fill"
    elif side == "buy" and limit_up_locked:
        reason = "limit_up_locked_no_buy_fill"
    elif side == "sell" and limit_down_locked:
        reason = "limit_down_locked_no_sell_fill"
    elif side == "sell" and entry_date is not None and trade_date <= entry_date:
        reason = "t_plus_one_not_sellable"
    elif available_volume is None:
        reason = "missing_volume_fail_closed"
    elif isinstance(available_volume, bool) or not isinstance(available_volume, int) or available_volume < 0:
        reason = "invalid_volume_fail_closed"
    if reason:
        return FillResult(False, side, 0, float(reference_price), None, 0, 0, 0, 0, reason)

    cap = math.floor((available_volume * max_participation) / lot_size) * lot_size
    filled_qty = min(quantity, cap)
    if filled_qty <= 0:
        return FillResult(False, side, 0, float(reference_price), None, 0, 0, 0, 0,
                          "insufficient_participation_volume")
    slip = costs.slippage_bps / 10000.0
    fill_price = float(reference_price) * (1 + slip if side == "buy" else 1 - slip)
    turnover = fill_price * filled_qty
    commission = max(costs.minimum_commission, turnover * costs.commission_bps / 10000.0)
    stamp = turnover * costs.sell_stamp_duty_bps / 10000.0 if side == "sell" else 0.0
    transfer = turnover * costs.transfer_fee_bps / 10000.0
    total = commission + stamp + transfer
    return FillResult(True, side, filled_qty, float(reference_price), round(fill_price, 4),
                      round(commission, 2), round(stamp, 2), round(transfer, 2),
                      round(total, 2), "filled" if filled_qty == quantity else "partial_fill_volume_cap")
