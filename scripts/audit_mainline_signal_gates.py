"""Run independent ignition evidence and lifecycle evidence audits."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.model.mainline_signal_gates import (
    audit_emotion_history,
    attach_startup_evidence,
    evaluate_peak_and_decay_independently,
    gate_ignition_labels,
)


def evidence_counts(mask: pd.Series) -> dict[str, int]:
    """Count boolean evidence without collapsing missing values into False."""
    return {
        "true": int(mask.eq(True).fillna(False).sum()),  # noqa: E712
        "false": int(mask.eq(False).fillna(False).sum()),  # noqa: E712
        "unknown": int(mask.isna().sum()),
    }


def main() -> None:
    output = Path("output")
    phases = pd.read_csv(output / "sw_mainline_phase_replay.csv")
    emotion = pd.read_csv(output / "sw_mainline_market_cycle_history.csv")
    policy = pd.read_json(
        "data/policy_catalysts/official_policy_events.jsonl", lines=True
    )
    # The current policy archive is a seed corpus, not a complete historical
    # archive. Keep policy evidence UNKNOWN until source coverage is certified.
    startup = attach_startup_evidence(
        phases, emotion, policy, policy_archive_complete=False
    )
    lifecycle = evaluate_peak_and_decay_independently(startup)
    # Legacy IGNITION remains a candidate until all three independent gates pass.
    lifecycle = gate_ignition_labels(lifecycle)
    ignition_candidate = lifecycle["mainline_phase_candidate"].eq("IGNITION")
    confirmed = lifecycle["startup_evidence_status"].eq("CONFIRMED").fillna(False)

    gate_fields = [
        "emotion_reversal_component",
        "policy_catalyst_component",
        "capital_turnover_share_positive",
        "capital_turnover_share_increasing",
        "capital_leadership_top_quintile",
        "capital_relative_strength_accelerating",
        "capital_concentration_component",
    ]
    positive_policy_ignition = (
        ignition_candidate & lifecycle["policy_catalyst_component"].eq(True)
    )
    ignition_gate_breakdown = {
        field: evidence_counts(lifecycle.loc[ignition_candidate, field])
        for field in gate_fields
    }
    positive_policy_gate_breakdown = {
        field: evidence_counts(lifecycle.loc[positive_policy_ignition, field])
        for field in gate_fields
    }
    lifecycle.to_csv(output / "sw_mainline_independent_evidence.csv", index=False)
    audit = {
        "emotion_data_availability": audit_emotion_history(emotion),
        "policy_archive_complete": False,
        "policy_event_rows": int(len(policy)),
        "startup_evidence_status_counts": {
            str(k): int(v) for k, v in startup.startup_evidence_status.value_counts(
                dropna=False
            ).items()
        },
        "peak_evidence_counts": {
            str(k): int(v) for k, v in lifecycle.peak_evidence.value_counts(
                dropna=False
            ).items()
        },
        "decay_evidence_counts": {
            str(k): int(v) for k, v in lifecycle.decay_evidence.value_counts(
                dropna=False
            ).items()
        },
        "gated_ignition_candidate_rows": int(ignition_candidate.sum()),
        "confirmed_ignition_rows": int((ignition_candidate & confirmed).sum()),
        "unconfirmed_ignition_rows": int((ignition_candidate & ~confirmed).sum()),
        "ignition_candidate_gate_breakdown": ignition_gate_breakdown,
        "positive_policy_ignition_rows": int(positive_policy_ignition.sum()),
        "positive_policy_ignition_gate_breakdown": positive_policy_gate_breakdown,
        "startup_live_eligible": False,
        "historical_pit_eligible": False,
    }
    (output / "sw_mainline_evidence_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False))


if __name__ == "__main__":
    main()
