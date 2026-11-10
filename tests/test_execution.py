from datetime import date
import pytest

from src.model.execution import ExecutionCosts, simulate_daily_fill


D1 = date(2026, 11, 10)
D2 = date(2026, 11, 11)


def call(**kwargs):
    base = dict(side="buy", quantity=1000, reference_price=10.0,
                trade_date=D1, available_volume=100000)
    base.update(kwargs)
    return simulate_daily_fill(**base)


def test_buy_fill_applies_slippage_and_costs():
    r = call()
    assert r.filled and r.quantity == 1000
    assert r.fill_price > r.reference_price
    assert r.total_cost >= 5


def test_sell_fill_applies_stamp_duty():
    r = call(side="sell", trade_date=D2, entry_date=D1)
    assert r.filled and r.stamp_duty > 0


def test_t_plus_one_blocks_same_day_sale():
    r = call(side="sell", entry_date=D1)
    assert not r.filled and r.reason == "t_plus_one_not_sellable"


def test_locked_limit_queues_fail_closed():
    assert call(limit_up_locked=True).reason == "limit_up_locked_no_buy_fill"
    assert call(side="sell", trade_date=D2, entry_date=D1,
                limit_down_locked=True).reason == "limit_down_locked_no_sell_fill"


def test_suspension_and_missing_volume_do_not_fill():
    assert call(suspended=True).quantity == 0
    assert call(available_volume=None).reason == "missing_volume_fail_closed"


def test_participation_cap_allows_partial_fill_only():
    r = call(available_volume=10000, max_participation=0.05)
    assert r.filled and r.quantity == 500
    assert r.reason == "partial_fill_volume_cap"


@pytest.mark.parametrize("quantity", [0, -100, 101, True])
def test_invalid_quantity_rejected(quantity):
    with pytest.raises(ValueError):
        call(quantity=quantity)


def test_invalid_costs_rejected():
    with pytest.raises(ValueError):
        call(costs=ExecutionCosts(slippage_bps=float("nan")))

@pytest.mark.parametrize("participation", [True, False, float("nan"), float("inf"), "0.5"])
def test_invalid_participation_is_rejected(participation):
    with pytest.raises(ValueError):
        call(max_participation=participation)


@pytest.mark.parametrize("lot_size", [True, False, 0, -100, 100.0])
def test_invalid_lot_size_is_rejected(lot_size):
    with pytest.raises(ValueError):
        call(lot_size=lot_size)


def test_execution_costs_reject_non_numeric_values_at_construction():
    with pytest.raises(ValueError):
        ExecutionCosts(slippage_bps="10")
