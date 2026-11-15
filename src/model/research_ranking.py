"""Research-only opportunity ranking, separate from formal M5 authorization.

Only same-session candle geometry, observed liquidity and current industry
context are used. No unadjusted cross-session return, missing-value imputation,
or formal buy signal is introduced.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


RESEARCH_FACTORS = {
    # Current quote factors keep the live research rank usable when historical
    # OHLC is absent. They are explicitly not formal M5/PIT factors.
    "industry_strength": ("industry_relative_strength", True),
    "current_momentum": ("涨跌幅%", True),
    "current_liquidity": ("当日成交额", True),
    "price_structure": ("price_structure", True),
    "body_strength": ("BodyStrength", True),
    "upper_shadow_quality": ("upper_shadow_ratio", False),
    "trade_quality": ("trade_quality", True),
}


def build_research_opportunity_ranking(
    watchlist: pd.DataFrame,
    m5_input: pd.DataFrame,
    *,
    min_observed_factors: int = 3,
    industry_context: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Rank current research opportunities with explicit per-row factor coverage.

    This ranking is deliberately not an M5 score: it is forward-only context,
    and cannot authorize an order. It excludes raw cross-session returns because
    corporate-action-adjusted PIT total-return history is not universally ready.
    """
    if isinstance(min_observed_factors, bool) or not isinstance(min_observed_factors, int):
        raise ValueError("min_observed_factors must be an integer")
    if not 1 <= min_observed_factors <= len(RESEARCH_FACTORS):
        raise ValueError("min_observed_factors is outside the available factor count")
    if watchlist is None or watchlist.empty:
        return pd.DataFrame(), {
            "status": "NO_RESEARCH_ROWS", "row_count": 0,
            "min_observed_factors": min_observed_factors,
            "factor_count": len(RESEARCH_FACTORS),
            "formal_m5_score": False, "buy_authorization": False,
            "historical_pit_eligible": False,
        }
    if m5_input is None or m5_input.empty or "symbol" not in m5_input:
        m5 = pd.DataFrame({"symbol": watchlist.get("代码", pd.Series(dtype=str)).astype(str)})
    else:
        m5 = m5_input.copy()
    base = watchlist.copy()
    if "代码" in base:
        base["symbol"] = base["代码"].astype(str).str.strip().str.zfill(6)
    elif "symbol" in base:
        base["symbol"] = base["symbol"].astype(str).str.strip().str.zfill(6)
    else:
        raise ValueError("watchlist requires 代码 or symbol")
    m5["symbol"] = m5["symbol"].astype(str).str.strip().str.zfill(6)
    if base["symbol"].duplicated().any() or m5["symbol"].duplicated().any():
        raise ValueError("duplicate symbol in research ranking inputs")
    out = base.merge(m5, on="symbol", how="left", suffixes=("", "_m5"), validate="one_to_one")
    if industry_context is not None and not industry_context.empty:
        if "symbol" not in industry_context:
            raise ValueError("industry_context requires symbol")
        context = industry_context.copy()
        context["symbol"] = context["symbol"].astype(str).str.strip().str.zfill(6)
        if context["symbol"].duplicated().any():
            raise ValueError("duplicate symbol in forward industry context")
        keep = [c for c in (
            "symbol", "industry_code", "current_industry_level1", "mainline_phase",
            "mainline_phase_gated", "strength_percentile", "startup_evidence_status",
            "membership_observed_at", "mainline_trade_date",
        ) if c in context.columns]
        context = context[keep].rename(columns={
            "industry_code": "live_industry_code",
            "current_industry_level1": "live_industry_level1",
            "mainline_phase": "live_mainline_phase",
            "mainline_phase_gated": "live_mainline_phase_gated",
            "strength_percentile": "live_industry_strength_percentile",
            "startup_evidence_status": "live_startup_evidence_status",
        })
        out = out.merge(context, on="symbol", how="left", validate="one_to_one")
        live_strength = pd.to_numeric(
            out["live_industry_strength_percentile"], errors="coerce"
        ) * 100.0
        if "industry_relative_strength" not in out.columns:
            out["industry_relative_strength"] = live_strength
        else:
            out["industry_relative_strength"] = pd.to_numeric(
                out["industry_relative_strength"], errors="coerce"
            ).combine_first(live_strength)

    ranked_columns = []
    available_factors = []
    for factor_name, (source_column, higher_is_better) in RESEARCH_FACTORS.items():
        if source_column not in out:
            out[source_column] = np.nan
        values = pd.to_numeric(out[source_column], errors="coerce")
        finite = values.where(np.isfinite(values))
        rank = finite.rank(method="average", pct=True, ascending=higher_is_better) * 100.0
        rank_column = factor_name + "_percentile"
        out[rank_column] = rank
        ranked_columns.append(rank_column)
        available_factors.append(factor_name)

    matrix = out[ranked_columns]
    out["research_factor_count"] = matrix.notna().sum(axis=1).astype(int)
    out["research_factor_coverage"] = out["research_factor_count"] / len(ranked_columns)
    # A short-line opportunity rank without a mapped industry/mainline anchor
    # is just a technical-momentum list. Do not label it an opportunity rank:
    # keep the diagnostics, but require industry strength plus the configured
    # number of observed factors before producing a ranked score.
    mainline_anchor_ready = out["industry_strength_percentile"].notna()
    out["research_mainline_anchor_ready"] = mainline_anchor_ready
    out["research_opportunity_score"] = (
        matrix.mean(axis=1).where(
            (out["research_factor_count"] >= min_observed_factors)
            & mainline_anchor_ready
        )
    )
    out["research_opportunity_rank"] = (
        out["research_opportunity_score"].rank(method="min", ascending=False).astype("Int64")
    )
    out["research_rank_status"] = np.select(
        [
            ~mainline_anchor_ready,
            out["research_opportunity_score"].notna(),
        ],
        [
            "MISSING_MAINLINE_CONTEXT",
            "RANKED_RESEARCH_ONLY",
        ],
        default="INSUFFICIENT_OBSERVED_FACTORS",
    )
    out["research_rank_scope"] = "current_run_live_research_pool"
    out["research_rank_factor_set"] = ",".join(available_factors)
    out["research_rank_uses_raw_cross_session_returns"] = False
    out["research_rank_historical_pit_eligible"] = False
    out["research_rank_formal_m5_score"] = False
    out["research_rank_buy_authorization"] = False
    out = out.sort_values(
        ["research_opportunity_score", "research_factor_count", "symbol"],
        ascending=[False, False, True], na_position="last",
    ).reset_index(drop=True)
    audit = {
        "status": (
            "READY_RESEARCH_ONLY"
            if out["research_opportunity_score"].notna().any()
            else "MAINLINE_CONTEXT_UNAVAILABLE"
            if not mainline_anchor_ready.any()
            else "INSUFFICIENT_RESEARCH_FACTOR_COVERAGE"
        ),
        "row_count": int(len(out)),
        "ranked_count": int(out["research_opportunity_score"].notna().sum()),
        "unranked_count": int(out["research_opportunity_score"].isna().sum()),
        "mainline_context_ready_count": int(mainline_anchor_ready.sum()),
        "missing_mainline_context_count": int((~mainline_anchor_ready).sum()),
        "min_observed_factors": min_observed_factors,
        "factor_count": len(RESEARCH_FACTORS),
        "factor_sources": {name: source for name, (source, _) in RESEARCH_FACTORS.items()},
        "uses_raw_cross_session_returns": False,
        "historical_pit_eligible": False,
        "formal_m5_score": False,
        "buy_authorization": False,
    }
    return out, audit
