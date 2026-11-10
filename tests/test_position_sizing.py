import pytest
from src.model.position_sizing import size_long_position


def test_sizing_respects_per_trade_risk_and_board_lot():
    r = size_long_position(entry_price=20, stop_price=19, available_cash=100000,
                           open_position_count=0, open_portfolio_risk=0)
    assert r.approved and r.shares % 100 == 0
    assert r.estimated_loss <= 2000 and r.cash_required <= 100000


def test_remaining_portfolio_risk_caps_second_position():
    r = size_long_position(entry_price=20, stop_price=19, available_cash=100000,
                           open_position_count=1, open_portfolio_risk=2500)
    assert r.approved and r.risk_budget == 500
    assert r.estimated_loss <= 500


def test_rejects_when_maximum_positions_reached():
    r = size_long_position(entry_price=20, stop_price=19, available_cash=100000,
                           open_position_count=2, open_portfolio_risk=0)
    assert not r.approved and r.shares == 0
    assert "maximum_open_positions_reached" in r.reasons


def test_rejects_invalid_stop():
    r = size_long_position(entry_price=20, stop_price=20, available_cash=100000,
                           open_position_count=0, open_portfolio_risk=0)
    assert not r.approved and "invalid_long_entry_or_stop" in r.reasons


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), True, None])
def test_rejects_invalid_numeric_inputs(bad):
    r = size_long_position(entry_price=bad, stop_price=19, available_cash=100000,
                           open_position_count=0, open_portfolio_risk=0)
    assert not r.approved and r.shares == 0


def test_cash_limit_is_applied_after_risk_limit():
    r = size_long_position(entry_price=10, stop_price=9, available_cash=1500,
                           open_position_count=0, open_portfolio_risk=0)
    assert r.approved and r.shares == 100
    assert r.cash_required == 1000


def test_sizing_reserves_fixed_minimum_round_trip_commissions():
    r = size_long_position(entry_price=10, stop_price=9, available_cash=100000,
                           open_position_count=0, open_portfolio_risk=0,
                           per_trade_risk_limit=2000)
    assert r.approved
    assert r.estimated_loss == pytest.approx(r.shares * (1 + 10 * 30 / 10000) + 10)
    assert r.estimated_loss <= 2000


def test_cash_limit_reserves_buy_fees_and_keeps_smaller_affordable_lot():
    r = size_long_position(entry_price=10, stop_price=9, available_cash=2005,
                           open_position_count=0, open_portfolio_risk=0,
                           per_trade_risk_limit=2000)
    assert r.approved
    assert r.shares == 100
    buy_fees = max(5.0, r.shares * r.entry_price * 2.5 / 10000) + r.shares * r.entry_price * 0.1 / 10000
    assert r.cash_required + buy_fees <= 2005
