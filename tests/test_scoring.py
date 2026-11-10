import pytest

from src.model.r21_core import A_WEIGHTS, B_WEIGHTS, C_WEIGHTS, MarketState
from src.model.scoring import candidate_band, score_a, score_b, score_c


def factors(weights):
    return {key: 80.0 for key in weights}


def all_true(keys):
    return {key: True for key in keys}


def test_a_score_preserves_original_weights_and_hard_gate_cannot_be_offset():
    gates = all_true([
        "industry_top_30pct", "stock_residual_top_20pct", "amount_ma20_ge_100m",
        "breakout_or_confirmed_retest", "no_high_volume_stall_or_repeated_rejection",
    ])
    result = score_a(factors(A_WEIGHTS), gates, market_state=MarketState.RISK_ON)
    assert result.score == pytest.approx(80)
    assert result.eligible
    gates["industry_top_30pct"] = False
    failed = score_a(factors(A_WEIGHTS), gates, market_state=MarketState.RISK_ON)
    assert failed.score == pytest.approx(80)
    assert not failed.eligible
    assert "hard_gate_failed:industry_top_30pct" in failed.reasons


def test_a_missing_factor_means_no_score():
    values = factors(A_WEIGHTS)
    values.pop("catalyst")
    result = score_a(values, {}, market_state=MarketState.RISK_ON)
    assert result.score is None
    assert not result.eligible


def test_risk_off_blocks_all_strategies_even_with_high_scores():
    a = score_a(factors(A_WEIGHTS), all_true([
        "industry_top_30pct", "stock_residual_top_20pct", "amount_ma20_ge_100m",
        "breakout_or_confirmed_retest", "no_high_volume_stall_or_repeated_rejection",
    ]), market_state=MarketState.RISK_OFF)
    assert a.score == pytest.approx(80)
    assert not a.eligible
    assert "market_risk_off" in a.reasons


def test_b_refuses_previous_close_lookahead_and_requires_next_day_fields():
    close_result = score_b(factors(B_WEIGHTS), {}, market_state=MarketState.NEUTRAL, stage="close")
    assert close_result.score is None
    assert not close_result.eligible
    gates = all_true(["event_data_verified", "event_subtype_supported",
                      "liquidity_and_risk_pass", "announcement_risk_pass"])
    next_day = score_b(factors(B_WEIGHTS), gates, market_state=MarketState.NEUTRAL, stage="next_day")
    assert next_day.score == pytest.approx(80)
    assert next_day.eligible


def test_c_requires_each_pullback_confirmation_gate():
    keys = ["pre_pullback_residual_positive", "pullback_between_0_5_and_4pct",
            "pullback_volume_le_80pct_prior5d", "stop_or_reversal_confirmed",
            "industry_not_weak", "no_material_negative_event", "liquidity_and_risk_pass"]
    gates = all_true(keys)
    ok = score_c(factors(C_WEIGHTS), gates, market_state=MarketState.NEUTRAL)
    assert ok.eligible and ok.score == pytest.approx(80)
    gates["pullback_volume_le_80pct_prior5d"] = False
    failed = score_c(factors(C_WEIGHTS), gates, market_state=MarketState.NEUTRAL)
    assert not failed.eligible
    assert failed.score == pytest.approx(80)


def test_candidate_bands_are_not_buy_signals():
    assert candidate_band(95) == "priority_research"
    assert candidate_band(85) == "candidate"
    assert candidate_band(75) == "watch"
    assert candidate_band(69) == "below_threshold"
    with pytest.raises(ValueError):
        candidate_band(101)
