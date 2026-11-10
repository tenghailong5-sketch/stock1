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
    gates = all_true(["event_data_verified",
                      "liquidity_and_risk_pass", "announcement_risk_pass"])
    gates["event_subtype"] = "close_up_sealed"
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


def test_factor_percentiles_are_scoped_to_each_date_and_preserve_missing():
    import pandas as pd
    from src.model.scoring import score_factor_bundle
    frame = pd.DataFrame({"trade_date": ["2026-01-01", "2026-01-01", "2026-01-02", "2026-01-02"], "raw": [10.0, 20.0, 100.0, None]})
    out = score_factor_bundle(frame, {"industry_relative_strength": ("raw", True)})
    assert out.loc[0, "industry_relative_strength"] == 50.0
    assert out.loc[1, "industry_relative_strength"] == 100.0
    assert out.loc[2, "industry_relative_strength"] == 100.0
    assert pd.isna(out.loc[3, "industry_relative_strength"])


def test_b_requires_one_of_four_distinct_event_subtypes():
    gates = {"event_data_verified": True, "liquidity_and_risk_pass": True, "announcement_risk_pass": True, "event_subtype": "unknown_event"}
    result = score_b(factors(B_WEIGHTS), gates, market_state=MarketState.NEUTRAL, stage="next_day")
    assert not result.eligible
    assert result.score is None


def test_boolean_values_are_not_accepted_as_numeric_factor_scores():
    values = factors(A_WEIGHTS)
    values["catalyst"] = True
    result = score_a(values, {}, market_state=MarketState.RISK_ON)
    assert result.score is None
    assert not result.eligible
    assert "missing_or_invalid_factor:catalyst" in result.reasons


def test_non_finite_factor_values_are_rejected():
    import math
    from src.model.r21_core import weighted_score
    for invalid in (float("nan"), float("inf"), float("-inf")):
        values = factors(A_WEIGHTS)
        values["catalyst"] = invalid
        result = score_a(values, {}, market_state=MarketState.RISK_ON)
        assert result.score is None
        with pytest.raises(ValueError):
            weighted_score(A_WEIGHTS, values)


def test_candidate_floor_is_hard_and_cannot_be_configured_below_80():
    from src.model.scoring import qualify_candidate
    gates = all_true([
        "industry_top_30pct", "stock_residual_top_20pct", "amount_ma20_ge_100m",
        "breakout_or_confirmed_retest", "no_high_volume_stall_or_repeated_rejection",
    ])
    result = score_a(factors(A_WEIGHTS), gates, market_state=MarketState.RISK_ON)
    assert qualify_candidate(result).eligible
    below = result.__class__(result.strategy, 79.99, True, (), result.stage)
    rejected = qualify_candidate(below)
    assert not rejected.eligible
    assert rejected.score == pytest.approx(79.99)
    assert "below_minimum_score:80" in rejected.reasons
    with pytest.raises(ValueError):
        qualify_candidate(result, minimum_score=79)


def test_candidate_floor_never_overrides_failed_hard_gate():
    from src.model.scoring import qualify_candidate
    gates = all_true([
        "industry_top_30pct", "stock_residual_top_20pct", "amount_ma20_ge_100m",
        "breakout_or_confirmed_retest", "no_high_volume_stall_or_repeated_rejection",
    ])
    gates["industry_top_30pct"] = False
    result = score_a(factors(A_WEIGHTS), gates, market_state=MarketState.RISK_ON)
    qualified = qualify_candidate(result)
    assert not qualified.eligible
    assert "hard_gate_failed:industry_top_30pct" in qualified.reasons
