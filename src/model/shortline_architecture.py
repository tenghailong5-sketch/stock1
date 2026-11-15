"""Compact five-pillar short-term strategy architecture audit.

This module consolidates the existing M3/M4/M5 inputs into five decision
pillars. It is an audit/diagnostic layer only: it does not invent missing
values, alter strategy weights, or grant trade eligibility.
"""
from __future__ import annotations

import math
from collections import Counter

import numpy as np
import pandas as pd


PILLARS = {
    "market_sentiment": {
        "factors": ("market_environment",),
        "gates": (),
    },
    "theme_mainline": {
        "factors": ("industry_relative_strength", "catalyst"),
        "gates": ("industry_top_30pct",),
    },
    "capital_concentration": {
        "factors": ("stock_residual_strength", "trade_quality"),
        "gates": ("stock_residual_top_20pct", "amount_ma20_ge_100m"),
    },
    "stock_structure": {
        "factors": (
            "price_structure", "pre_pullback_strength",
            "pullback_reversal_structure",
        ),
        "gates": (
            "breakout_or_confirmed_retest",
            "pre_pullback_residual_positive",
            "pullback_between_0_5_and_4pct",
            "pullback_volume_le_80pct_prior5d",
            "stop_or_reversal_confirmed",
        ),
    },
    "execution_risk": {
        "factors": (),
        "gates": (
            "no_high_volume_stall_or_repeated_rejection",
            "industry_not_weak",
            "no_material_negative_event",
            "liquidity_and_risk_pass",
        ),
    },
}

# Aliases are descriptive only and are never counted as additional factors.
FACTOR_ALIASES = {
    "industry_strength": "industry_relative_strength",
}


def _valid_factor(value) -> bool:
    if isinstance(value, (bool, type(None))):
        return False
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return False
    return math.isfinite(number) and 0.0 <= number <= 100.0


def _valid_gate(value) -> bool:
    return isinstance(value, (bool, np.bool_))


def audit_shortline_architecture(frame: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """Audit five-pillar coverage per symbol without changing M5 scores."""
    if frame is None:
        frame = pd.DataFrame()
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")

    records = []
    pillar_counts = {name: Counter() for name in PILLARS}
    symbols = frame.to_dict(orient="records")
    for row in symbols:
        symbol = str(row.get("symbol", "")).strip()
        record = {"symbol": symbol}
        for pillar, spec in PILLARS.items():
            required = list(spec["factors"]) + list(spec["gates"])
            valid = 0
            for field in spec["factors"]:
                if _valid_factor(row.get(field)):
                    valid += 1
            for field in spec["gates"]:
                if _valid_gate(row.get(field)):
                    valid += 1
            missing = len(required) - valid
            status = "READY" if required and missing == 0 else (
                "MISSING" if valid == 0 else "PARTIAL"
            )
            record[pillar + "_status"] = status
            record[pillar + "_available"] = valid
            record[pillar + "_required"] = len(required)
            pillar_counts[pillar][status] += 1
        record["pit_backed"] = (
            row.get("m5_pit_factor_provenance") == "verified_pit_snapshot"
        )
        record["diagnostic_only"] = not record["pit_backed"]
        records.append(record)

    details = pd.DataFrame(records)
    summary = {
        "architecture": "five_pillar_shortline_v1",
        "status": "AUDITED" if len(details) else "NO_CANDIDATES",
        "candidate_count": int(len(details)),
        "pit_backed_candidate_count": int(
            details["pit_backed"].sum()
        ) if not details.empty else 0,
        "diagnostic_only_candidate_count": int(
            details["diagnostic_only"].sum()
        ) if not details.empty else 0,
        "duplicate_aliases_excluded": dict(FACTOR_ALIASES),
        "pillars": {
            name: {
                "required_inputs": len(spec["factors"]) + len(spec["gates"]),
                "ready": int(pillar_counts[name]["READY"]),
                "partial": int(pillar_counts[name]["PARTIAL"]),
                "missing": int(pillar_counts[name]["MISSING"]),
                "factors": list(spec["factors"]),
                "gates": list(spec["gates"]),
            }
            for name, spec in PILLARS.items()
        },
        "score_weights_changed": False,
        "trade_authorization_changed": False,
    }
    return summary, details
