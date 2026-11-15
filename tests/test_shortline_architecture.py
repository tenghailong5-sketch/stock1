import pandas as pd

from src.model.shortline_architecture import audit_shortline_architecture


def test_five_pillar_audit_counts_core_inputs_without_double_counting_aliases():
    frame = pd.DataFrame([{
        "symbol": "000001",
        "market_environment": 80,
        "industry_relative_strength": 90,
        "industry_strength": 90,
        "catalyst": 70,
        "industry_top_30pct": True,
        "stock_residual_strength": 85,
        "trade_quality": 75,
        "stock_residual_top_20pct": True,
        "amount_ma20_ge_100m": True,
        "price_structure": 80,
        "pre_pullback_strength": 70,
        "pullback_reversal_structure": 65,
        "breakout_or_confirmed_retest": True,
        "pre_pullback_residual_positive": True,
        "pullback_between_0_5_and_4pct": True,
        "pullback_volume_le_80pct_prior5d": True,
        "stop_or_reversal_confirmed": True,
        "no_high_volume_stall_or_repeated_rejection": True,
        "industry_not_weak": True,
        "no_material_negative_event": True,
        "liquidity_and_risk_pass": True,
        "m5_pit_factor_provenance": "verified_pit_snapshot",
    }])
    summary, details = audit_shortline_architecture(frame)

    assert summary["architecture"] == "five_pillar_shortline_v1"
    assert summary["candidate_count"] == 1
    assert summary["pit_backed_candidate_count"] == 1
    assert summary["pillars"]["theme_mainline"]["required_inputs"] == 3
    assert details.loc[0, "theme_mainline_status"] == "READY"
    assert details.loc[0, "capital_concentration_status"] == "READY"
    assert details.loc[0, "stock_structure_status"] == "READY"
    assert details.loc[0, "execution_risk_status"] == "READY"
    assert summary["score_weights_changed"] is False


def test_missing_inputs_remain_partial_or_missing_and_do_not_gain_authority():
    frame = pd.DataFrame([{
        "symbol": "000002",
        "market_environment": 75,
        "industry_relative_strength": 88,
        "m5_pit_factor_provenance": "missing_or_ineligible_pit_snapshot",
    }])
    summary, details = audit_shortline_architecture(frame)

    assert details.loc[0, "market_sentiment_status"] == "READY"
    assert details.loc[0, "theme_mainline_status"] == "PARTIAL"
    assert details.loc[0, "capital_concentration_status"] == "MISSING"
    assert details.loc[0, "execution_risk_status"] == "MISSING"
    assert summary["diagnostic_only_candidate_count"] == 1
    assert summary["trade_authorization_changed"] is False


def test_empty_input_returns_explicit_no_candidates():
    summary, details = audit_shortline_architecture(pd.DataFrame())
    assert summary["status"] == "NO_CANDIDATES"
    assert summary["candidate_count"] == 0
    assert details.empty
