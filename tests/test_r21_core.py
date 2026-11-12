from datetime import datetime, timezone, timedelta
import pytest
from src.model.r21_core import (
    A_WEIGHTS, B_WEIGHTS, C_WEIGHTS, ExitState, MarketState, PositionExit,
    can_open_position, conservative_daily_exit, point_in_time_rows,
    position_risk_budget, weighted_score,
)


def test_strategy_weights_are_original_and_sum_to_100():
    assert sum(A_WEIGHTS.values()) == 100
    assert sum(B_WEIGHTS.values()) == 100
    assert sum(C_WEIGHTS.values()) == 100


def test_score_refuses_missing_factor_instead_of_imputing_zero():
    with pytest.raises(ValueError):
        weighted_score(A_WEIGHTS, {"industry_relative_strength": 80})


@pytest.mark.parametrize("bad_weight", [float("nan"), float("inf"), True, "20"])
def test_weight_validation_rejects_non_numeric_or_non_finite_values(bad_weight):
    with pytest.raises(ValueError, match="finite non-negative numeric"):
        weighted_score({"a": bad_weight, "b": 100}, {"a": 50, "b": 50})


def test_point_in_time_excludes_future_and_unknown_availability():
    now = datetime(2026, 11, 10, 8, tzinfo=timezone.utc)
    rows = [
        {"available_time": now - timedelta(minutes=1), "v": 1},
        {"available_time": now + timedelta(minutes=1), "v": 2},
        {"available_time": None, "v": 3},
    ]
    assert [r["v"] for r in point_in_time_rows(rows, now)] == [1]


def test_risk_budget_never_exceeds_portfolio_cap():
    first = position_risk_budget(1)
    second = position_risk_budget(2, first_used=first, same_industry_or_corr_gt_06=True)
    assert first == 2000
    assert second == 500 or second == 0  # below 600 threshold is rejected
    assert first + second <= 3000


def test_risk_off_or_missing_inputs_blocks_new_position():
    assert not can_open_position(a_data_valid=True, ruleset_known=True,
        corporate_actions_resolved=True, risk_inputs_valid=True,
        market=MarketState.RISK_OFF, prediction_available=True)
    assert not can_open_position(a_data_valid=True, ruleset_known=True,
        corporate_actions_resolved=False, risk_inputs_valid=True,
        market=MarketState.RISK_ON, prediction_available=True)


def test_daily_double_touch_assumes_stop_first():
    assert conservative_daily_exit(10, 11.2, 9.3, 11, 9.5) == "STOP_FIRST"


def test_blocked_exit_does_not_reduce_position():
    p = PositionExit(remaining_qty=100)
    p.trigger()
    p.blocked(ExitState.LIMIT_DOWN_LOCKED)
    assert p.remaining_qty == 100
    assert p.state == ExitState.LIMIT_DOWN_LOCKED


def test_partial_exit_reduces_only_filled_quantity():
    p = PositionExit(remaining_qty=100)
    p.trigger()
    p.apply_fill(40)
    assert p.remaining_qty == 60
    assert p.state == ExitState.PARTIALLY_FILLED
