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
        "available_at": ["2026-01-02T08:00:00Z", "2026-01-02T08:00:00Z"],
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
