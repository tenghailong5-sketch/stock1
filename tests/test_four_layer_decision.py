import pytest

from src.model.four_layer_decision import (
    evaluate_trade_decision,
    industry_permission,
    market_permission,
)


def test_market_state_controls_entry_permission_and_risk_scale():
    assert market_permission("RISK_ON", 1.0, data_ready=True).mode == "ATTACK"
    neutral = market_permission("NEUTRAL", 0.7, data_ready=True)
    assert neutral.mode == "PROBE"
    assert neutral.risk_scale == 0.7
    assert not market_permission("RISK_OFF", 0.0, data_ready=True).allow_new_entries
    assert not market_permission("RISK_ON", 1.0, data_ready=False).allow_new_entries


def test_market_state_scale_mismatch_fails_closed():
    result = market_permission("RISK_ON", 0.7, data_ready=True)
    assert result.mode == "BLOCKED"
    assert not result.allow_new_entries


def test_industry_ignition_requires_pit_and_complete_stock_turnover():
    ready = industry_permission(
        "IGNITION", persistence_ready=True, historical_pit_eligible=True,
        stock_turnover_pit_status="ready",
    )
    assert ready.allow_new_entries
    assert not industry_permission(
        "IGNITION", persistence_ready=True, historical_pit_eligible=False,
        stock_turnover_pit_status="ready",
    ).allow_new_entries
    assert not industry_permission(
        "IGNITION", persistence_ready=True, historical_pit_eligible=True,
        stock_turnover_pit_status="incomplete_industry_coverage",
    ).allow_new_entries


def test_peak_is_hold_only_and_decay_blocks_new_entries():
    args = dict(
        persistence_ready=True, historical_pit_eligible=True,
        stock_turnover_pit_status="ready",
    )
    assert industry_permission("PEAK", **args).mode == "HOLD_ONLY"
    assert not industry_permission("PEAK", **args).allow_new_entries
    assert industry_permission("DECAY", **args).mode == "EXIT_OR_DEFEND"


def valid_trade(**overrides):
    args = dict(
        market_state="RISK_ON", market_position_scale=1.0, market_data_ready=True,
        mainline_phase="IGNITION", industry_persistence_ready=True,
        industry_historical_pit_eligible=True, stock_turnover_pit_status="ready",
        strategy_eligible=True, strategy_score=84.0, entry_ready=True,
        entry_price=20.0, stop_price=19.0, quantity=1000,
        planned_risk=1100.0, reward_risk=2.0, minimum_reward_risk=1.5,
        existing_portfolio_risk=0.0, existing_position_count=0,
    )
    args.update(overrides)
    return evaluate_trade_decision(**args)


def test_valid_four_layer_trade_passes_all_gates():
    result = valid_trade()
    assert result.allowed
    assert result.max_allowed_risk == 2000.0
    assert result.reasons == ()


@pytest.mark.parametrize("overrides,reason", [
    ({"market_state": "RISK_OFF", "market_position_scale": 0.0}, "confirmed_risk_off"),
    ({"mainline_phase": "PEAK"}, "mainline_peak_no_fresh_entry"),
    ({"mainline_phase": "DECAY"}, "mainline_decay"),
    ({"strategy_score": 79.9}, "M5_score_missing_or_below_80"),
    ({"entry_ready": False}, "entry_not_confirmed"),
    ({"planned_risk": 2001.0}, "planned_risk_exceeds_market_or_portfolio_budget"),
    ({"reward_risk": 1.0}, "reward_risk_below_configured_minimum"),
    ({"existing_position_count": 2}, "maximum_two_positions_reached_or_invalid_count"),
])
def test_four_layer_gate_blocks_invalid_trade(overrides, reason):
    result = valid_trade(**overrides)
    assert not result.allowed
    assert reason in result.reasons


def test_neutral_market_scales_risk_budget_without_changing_hard_caps():
    result = valid_trade(
        market_state="NEUTRAL", market_position_scale=0.7, planned_risk=1400.0
    )
    assert result.allowed
    assert result.max_allowed_risk == 1400.0


def test_second_position_respects_aggregate_3000_and_correlation_budget():
    result = valid_trade(
        existing_position_count=1, existing_portfolio_risk=2000.0,
        planned_risk=900.0, same_industry_or_corr_gt_06=False,
    )
    assert result.allowed
    assert result.max_allowed_risk == 1000.0
    correlated = valid_trade(
        existing_position_count=1, existing_portfolio_risk=2000.0,
        planned_risk=600.0, same_industry_or_corr_gt_06=True,
    )
    assert not correlated.allowed
    assert correlated.max_allowed_risk == 500.0


def test_planned_risk_cannot_understate_entry_to_stop_loss():
    result = valid_trade(planned_risk=900.0)
    assert not result.allowed
    assert "planned_risk_understates_gross_stop_risk" in result.reasons
