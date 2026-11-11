"""Research-only cross-sectional ranks for observable M4 price primitives.

These percentiles are not the composite FHM price-structure or trade-quality
scores. They remain observation-only because the model does not specify a
validated aggregation formula and the current provider snapshot is not
historical point-in-time eligible.
"""
from __future__ import annotations

import pandas as pd

from .m4_percentile import audited_percentile_rank


# Only directions that are explicit in the baseline definitions are ranked.
# Pullback, turnover/amount expansion and volume ratios are deliberately
# excluded: their desirability is conditional on trend and confirmation.
PRICE_OBSERVATION_MAP = {
    "CLV": ("CLV", True),
    "BodyStrength": ("BodyStrength", True),
    "return_5": ("return_5", True),
    "return_10": ("return_10", True),
    "return_20": ("return_20", True),
    "breakout20": ("breakout20", True),
    "upper_shadow_ratio": ("upper_shadow_ratio", False),
}


def rank_price_observations(
    frame: pd.DataFrame,
    *,
    expected_universe_size: int,
    scope_id: str = "top_350_turnover_snapshot",
    min_coverage: float = 0.95,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Attach audited per-date percentiles; incomplete dates remain all-NaN."""
    out = frame.copy()
    audits = []
    for output_col, (raw_col, higher_is_better) in PRICE_OBSERVATION_MAP.items():
        if raw_col not in out.columns:
            out[output_col + "_pct"] = float("nan")
            audits.append(pd.DataFrame([{
                "trade_date": None,
                "scope_id": scope_id,
                "factor": output_col,
                "expected_universe_size": expected_universe_size,
                "observed_universe_size": 0,
                "symbol_membership_sha256": None,
                "finite_factor_count": 0,
                "factor_coverage": 0.0,
                "min_coverage": min_coverage,
                "status": "missing_raw_factor",
            }]))
            continue
        ranks, audit = audited_percentile_rank(
            out,
            raw_col,
            scope_id=scope_id,
            expected_universe_size=expected_universe_size,
            higher_is_better=higher_is_better,
            min_coverage=min_coverage,
        )
        out[output_col + "_pct"] = ranks
        audit.insert(1, "factor", output_col)
        audits.append(audit)
    audit_frame = pd.concat(audits, ignore_index=True) if audits else pd.DataFrame()
    return out, audit_frame
