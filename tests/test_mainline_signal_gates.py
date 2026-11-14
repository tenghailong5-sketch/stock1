import pandas as pd

from src.model.mainline_signal_gates import (
    audit_emotion_history,
    attach_startup_evidence,
    evaluate_peak_and_decay_independently,
)


def test_emotion_audit_reports_missing_fields_without_zero_fill():
    frame = pd.DataFrame({
        "trade_date": ["2026-01-01", "2026-01-02"],
        "breadth": [0.2, 0.3],
        "limit_up_count": [3, 4],
    })
    audit = audit_emotion_history(frame)
    assert audit["fields"]["breadth"]["coverage"] == 1.0
    assert audit["fields"]["promotion_rate"]["status"] == "MISSING_FIELD"
    assert audit["missing_values_imputed"] == 0
    assert audit["zero_filled"] == 0


def test_only_positive_policy_match_is_known_under_incomplete_archive():
    industry = pd.DataFrame({
        "trade_date": ["2026-01-02", "2026-01-02", "2026-01-02"],
        "industry_code": ["I1", "I2", "I3"],
        "turnover_share": [0.12, 0.12, 0.12],
        "turnover_share_change_3d": [0.03, 0.03, 0.03],
        "strength_3d_percentile": [0.95, 0.95, 0.95],
        "relative_strength_change_2d": [0.02, 0.02, 0.02],
    })
    emotion = pd.DataFrame({
        "trade_date": ["2026-01-01", "2026-01-02"],
        "market_cycle": ["ICE", "RECOVERY"],
    })
    policy = pd.DataFrame({
        "available_at": ["2026-01-02T06:00:00Z", "2026-01-02T06:00:00Z"],
        "industry_codes": [["I1"], ["I2"]],
        "impact_direction": ["positive", "mixed"],
    })
    partial = attach_startup_evidence(industry, emotion, policy)
    assert partial.policy_catalyst_component.iloc[0] == True
    assert pd.isna(partial.policy_catalyst_component.iloc[1])
    assert pd.isna(partial.policy_catalyst_component.iloc[2])
    assert partial.startup_evidence_status.tolist() == ["CONFIRMED", "UNKNOWN", "UNKNOWN"]
    complete = attach_startup_evidence(
        industry, emotion, policy, policy_archive_complete=True
    )
    assert complete.policy_catalyst_component.tolist() == [True, False, False]
    assert complete.startup_evidence_status.tolist() == [
        "CONFIRMED", "NOT_CONFIRMED", "NOT_CONFIRMED"
    ]
    assert not partial.startup_live_eligible.any()


def test_capital_gate_exposes_each_condition_without_relaxing_composite():
    industry = pd.DataFrame({
        "trade_date": ["2026-01-02"],
        "industry_code": ["I1"],
        "turnover_share": [0.04],
        "turnover_share_change_3d": [0.01],
        "strength_3d_percentile": [0.60],
        "relative_strength_change_2d": [0.02],
    })
    emotion = pd.DataFrame({
        "trade_date": ["2026-01-01", "2026-01-02"],
        "market_cycle": ["ICE", "RECOVERY"],
    })
    policy = pd.DataFrame({
        "available_at": ["2026-01-02T06:00:00Z"],
        "industry_codes": [["I1"]],
        "impact_direction": ["positive"],
    })
    out = attach_startup_evidence(industry, emotion, policy)
    assert out.capital_turnover_share_positive.iloc[0]
    assert out.capital_turnover_share_increasing.iloc[0]
    assert not out.capital_leadership_at_ignition_threshold.iloc[0]
    assert out.capital_relative_strength_accelerating.iloc[0]
    assert not out.capital_concentration_component.iloc[0]
    assert out.startup_evidence_status.iloc[0] == "NOT_CONFIRMED"


def test_peak_and_decay_are_independent_and_missing_stays_unknown():
    frame = pd.DataFrame({
        "strength_percentile": [0.95, 0.2, None],
        "turnover_share_percentile": [0.9, 0.2, 0.9],
        "turnover_share_change_3d": [-0.01, -0.03, 0.01],
        "relative_strength_change_3d": [-0.01, -0.03, 0.01],
        "relative_strength_5d": [0.05, -0.03, 0.01],
    })
    out = evaluate_peak_and_decay_independently(frame)
    assert out.peak_evidence.tolist()[:2] == [True, False]
    assert out.decay_evidence.tolist()[:2] == [False, True]
    assert pd.isna(out.peak_evidence.iloc[2])
    assert pd.isna(out.decay_evidence.iloc[2])


def test_ignition_candidate_is_not_confirmed_when_any_gate_is_unknown():
    from src.model.mainline_signal_gates import gate_ignition_labels

    frame = pd.DataFrame({
        "mainline_phase": ["IGNITION", "IGNITION", "DECAY"],
        "startup_evidence_status": ["UNKNOWN", "CONFIRMED", "UNKNOWN"],
    })
    out = gate_ignition_labels(frame)
    assert out.mainline_phase_gated.tolist() == [
        "IGNITION_UNCONFIRMED", "IGNITION", "DECAY"
    ]
    assert not out.startup_live_eligible.any()


def test_evidence_counts_preserves_unknown_values():
    from scripts.audit_mainline_signal_gates import evidence_counts

    values = pd.Series([True, False, pd.NA, True, pd.NA], dtype="boolean")
    assert evidence_counts(values) == {"true": 2, "false": 1, "unknown": 2}


def test_evidence_counts_handles_empty_series():
    from scripts.audit_mainline_signal_gates import evidence_counts

    assert evidence_counts(pd.Series([], dtype="boolean")) == {
        "true": 0, "false": 0, "unknown": 0
    }



def test_policy_event_uses_first_trading_close_after_weekend_or_post_close():
    industry = pd.DataFrame({
        "trade_date": [
            "2026-01-02", "2026-01-02",
            "2026-01-05", "2026-01-05",
        ],
        "industry_code": ["I1", "I2", "I1", "I2"],
        "turnover_share": [0.1, 0.1, 0.1, 0.1],
        "turnover_share_change_3d": [0.01, 0.01, 0.01, 0.01],
        "strength_3d_percentile": [0.9, 0.9, 0.9, 0.9],
        "relative_strength_change_2d": [0.02, 0.02, 0.02, 0.02],
    })
    emotion = pd.DataFrame({
        "trade_date": ["2026-01-01", "2026-01-02", "2026-01-05"],
        "market_cycle": ["ICE", "RECOVERY", "RECOVERY"],
    })
    policy = pd.DataFrame({
        # Friday 16:00 Shanghai: after the close, so effective Monday.
        # Saturday 10:00 Shanghai: weekend publication, also effective Monday.
        "available_at": [
            "2026-01-02T08:00:00Z",
            "2026-01-03T02:00:00Z",
        ],
        "industry_codes": [["I2"], ["I1"]],
        "impact_direction": ["positive", "positive"],
    })
    out = attach_startup_evidence(
        industry, emotion, policy, policy_archive_complete=True
    )
    friday = out[out.trade_date.eq(pd.Timestamp("2026-01-02"))]
    monday = out[out.trade_date.eq(pd.Timestamp("2026-01-05"))]
    assert friday.policy_catalyst_component.tolist() == [False, False]
    assert monday.policy_catalyst_component.tolist() == [True, True]


def test_capital_leadership_gate_matches_default_ignition_strength_threshold():
    industry = pd.DataFrame({
        "trade_date": ["2026-01-02"],
        "industry_code": ["I1"],
        "turnover_share": [0.04],
        "turnover_share_change_3d": [0.01],
        "strength_3d_percentile": [0.677],
        "relative_strength_change_2d": [0.02],
    })
    emotion = pd.DataFrame({
        "trade_date": ["2026-01-01", "2026-01-02"],
        "market_cycle": ["ICE", "RECOVERY"],
    })
    policy = pd.DataFrame({
        "available_at": ["2026-01-02T06:00:00Z"],
        "industry_codes": [["I1"]],
        "impact_direction": ["positive"],
    })
    out = attach_startup_evidence(industry, emotion, policy)
    assert out.capital_leadership_at_ignition_threshold.iloc[0]
    assert out.capital_concentration_component.iloc[0]
    assert out.startup_evidence_status.iloc[0] == "CONFIRMED"


def test_ignition_gate_emits_delayed_confirmation_without_backdating():
    frame = pd.DataFrame({
        "trade_date": [
            "2026-01-01", "2026-01-02", "2026-01-05",
            "2026-01-06", "2026-01-07",
        ],
        "industry_code": ["I1"] * 5,
        "mainline_phase": [
            "TRANSITION", "IGNITION", "IGNITION", "IGNITION", "DECAY"
        ],
        "phase_transition": [False, True, False, False, True],
        "startup_evidence_status": [
            "UNKNOWN", "NOT_CONFIRMED", "NOT_CONFIRMED",
            "CONFIRMED", "UNKNOWN",
        ],
    })
    from src.model.mainline_signal_gates import gate_ignition_labels

    out = gate_ignition_labels(frame)
    assert out.mainline_phase_gated.tolist() == [
        "TRANSITION", "IGNITION_UNCONFIRMED", "IGNITION_UNCONFIRMED",
        "IGNITION", "DECAY",
    ]
    assert out.phase_transition_gated.tolist() == [False, False, False, True, True]
    assert out.phase_gate_reason.iloc[3] == "ignition_confirmed_after_evidence_arrived"
    assert not out.startup_live_eligible.any()
