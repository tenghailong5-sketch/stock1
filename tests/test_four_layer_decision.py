import pytest

from src.model.four_layer_decision import (
    evaluate_trade_decision,
    industry_permission,
    market_permission,
)


def test_market_state_uses_dynamic_scale_without_fixed_70_percent_probe():
    assert market_permission("RISK_ON", 0.83, data_ready=True).mode == "ATTACK"
    neutral = market_permission("NEUTRAL", 0.43, data_ready=True)
    assert neutral.mode == "PROBE"
    assert neutral.risk_scale == 0.43
    assert not market_permission("RISK_OFF", 0.0, data_ready=True).allow_new_entries
    assert not market_permission("RISK_ON", 1.0, data_ready=False).allow_new_entries


def test_market_state_scale_mismatch_and_zero_neutral_scale_block():
    assert not market_permission("RISK_ON", 0.0, data_ready=True).allow_new_entries
    assert not market_permission("NEUTRAL", 0.0, data_ready=True).allow_new_entries
    assert not market_permission("UNKNOWN", 0.5, data_ready=True).allow_new_entries


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
        available_risk_budget=5000.0,
    )
    args.update(overrides)
    return evaluate_trade_decision(**args)


def test_valid_four_layer_trade_passes_all_gates():
    result = valid_trade()
    assert result.allowed
    assert result.max_allowed_risk == 5000.0
    assert result.reasons == ()


@pytest.mark.parametrize("overrides,reason", [
    ({"market_state": "RISK_OFF", "market_position_scale": 0.0}, "confirmed_risk_off"),
    ({"mainline_phase": "PEAK"}, "mainline_peak_no_fresh_entry"),
    ({"mainline_phase": "DECAY"}, "mainline_decay"),
    ({"strategy_score": None}, "M5_score_missing_or_out_of_range"),
    ({"entry_ready": False}, "entry_not_confirmed"),
    ({"planned_risk": 6000.0}, "planned_risk_exceeds_dynamic_available_budget"),
    ({"reward_risk": 1.0}, "reward_risk_below_configured_minimum"),
    ({"available_risk_budget": None}, "dynamic_available_risk_budget_missing_or_invalid"),
])
def test_four_layer_gate_blocks_invalid_trade(overrides, reason):
    result = valid_trade(**overrides)
    assert not result.allowed
    assert reason in result.reasons


def test_sub_80_score_is_not_rejected_by_a_fixed_cutoff():
    result = valid_trade(strategy_score=62.5)
    assert result.allowed


def test_position_count_and_portfolio_risk_have_no_fixed_caps():
    result = valid_trade(
        existing_position_count=12,
        existing_portfolio_risk=250000.0,
        planned_risk=4000.0,
        available_risk_budget=5000.0,
    )
    assert result.allowed
    assert result.max_allowed_risk == 5000.0


def test_dynamic_market_scale_scales_available_risk_budget():
    result = valid_trade(
        market_state="NEUTRAL", market_position_scale=0.43,
        planned_risk=2100.0, available_risk_budget=5000.0,
    )
    assert result.allowed
    assert result.max_allowed_risk == pytest.approx(2150.0)


def test_planned_risk_cannot_understate_entry_to_stop_loss():
    result = valid_trade(planned_risk=900.0)
    assert not result.allowed
    assert "planned_risk_understates_gross_stop_risk" in result.reasons


def test_live_forward_mode_requires_snapshot_and_m3_persistence():
    live = industry_permission(
        "IGNITION", persistence_ready=True, historical_pit_eligible=False,
        stock_turnover_pit_status="not_ready", decision_mode="live_forward",
        industry_live_data_eligible=True, stock_turnover_live_status="ready",
    )
    assert live.allow_new_entries

    # A fresh snapshot cannot bypass M3's industry-persistence requirement.
    not_persistent = industry_permission(
        "IGNITION", persistence_ready=False, historical_pit_eligible=False,
        stock_turnover_pit_status="not_ready", decision_mode="live_forward",
        industry_live_data_eligible=True, stock_turnover_live_status="ready",
    )
    assert not not_persistent.allow_new_entries
    assert not_persistent.reason == "industry_persistence_not_ready"

    # Missing stock-level turnover concentration remains diagnostic in live
    # mode once the live snapshot and industry persistence are both ready.
    assert industry_permission(
        "IGNITION", persistence_ready=True, historical_pit_eligible=False,
        stock_turnover_pit_status="not_ready", decision_mode="live_forward",
        industry_live_data_eligible=True, stock_turnover_live_status="not_ready",
    ).allow_new_entries

    assert not industry_permission(
        "IGNITION", persistence_ready=True, historical_pit_eligible=False,
        stock_turnover_pit_status="not_ready", decision_mode="live_forward",
        industry_live_data_eligible=False, stock_turnover_live_status="ready",
    ).allow_new_entries

    # Retrieval-time live evidence must never satisfy the historical path.
    assert not industry_permission(
        "IGNITION", persistence_ready=True, historical_pit_eligible=False,
        stock_turnover_pit_status="ready", decision_mode="historical_pit",
        industry_live_data_eligible=True, stock_turnover_live_status="ready",
    ).allow_new_entries
