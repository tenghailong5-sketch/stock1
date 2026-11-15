"""Research-only opportunity ranking, separate from formal M5 authorization.

Only same-session candle geometry, observed liquidity and current industry
context are used. No unadjusted cross-session return, missing-value imputation,
or formal buy signal is introduced.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


RESEARCH_FACTORS = {
    # Weights are renormalized across observed factors for each row.
    "industry_strength": ("industry_relative_strength", True, 0.25),
    "current_momentum": ("涨跌幅%", True, 0.30),
    "current_liquidity": ("当日成交额", True, 0.15),
    "industry_flow_concentration": ("industry_turnover_share_change_3d", True, 0.15),
    "mainline_lifecycle": ("live_mainline_phase_score", True, 0.10),
    "policy_catalyst": ("live_policy_catalyst_score", True, 0.05),
    "price_structure": ("price_structure", True, 0.0),
    "body_strength": ("BodyStrength", True, 0.0),
    "upper_shadow_quality": ("upper_shadow_ratio", False, 0.0),
    "trade_quality": ("trade_quality", True, 0.0),
}


def build_research_opportunity_ranking(
    watchlist: pd.DataFrame,
    m5_input: pd.DataFrame,
    *,
    min_observed_factors: int = 3,
    industry_context: pd.DataFrame | None = None,
    allow_quote_only_fallback: bool = False,
) -> tuple[pd.DataFrame, dict]:
    """Rank current research opportunities with explicit per-row factor coverage.

    This ranking is deliberately not an M5 score: it is forward-only context,
    and cannot authorize an order. It excludes raw cross-session returns because
    corporate-action-adjusted PIT total-return history is not universally ready.
    """
    if not isinstance(allow_quote_only_fallback, bool):
        raise ValueError("allow_quote_only_fallback must be boolean")
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
            "mainline_phase_gated", "strength_percentile", "turnover_share",
            "turnover_share_change_1d", "turnover_share_change_3d",
            "startup_evidence_status", "startup_policy_status",
            "peak_evidence", "decay_evidence", "phase_gate_reason",
            "membership_observed_at", "mainline_trade_date",
            "live_catalyst_candidate_count", "live_event_price_in_state",
            "live_event_price_in_reason", "live_event_pre_discovery_excess_20d",
            "live_event_post_discovery_excess_3d",
        ) if c in context.columns]
        context = context[keep].rename(columns={
            "industry_code": "live_industry_code",
            "current_industry_level1": "live_industry_level1",
            "mainline_phase": "live_mainline_phase",
            "mainline_phase_gated": "live_mainline_phase_gated",
            "strength_percentile": "live_industry_strength_percentile",
            "turnover_share": "live_industry_turnover_share",
            "turnover_share_change_1d": "live_industry_turnover_share_change_1d",
            "turnover_share_change_3d": "live_industry_turnover_share_change_3d",
            "startup_evidence_status": "live_startup_evidence_status",
            "startup_policy_status": "live_startup_policy_status",
        })
        out = out.merge(context, on="symbol", how="left", validate="one_to_one")
        live_flow_change = pd.to_numeric(
            out.get("live_industry_turnover_share_change_3d", pd.Series(np.nan, index=out.index)),
            errors="coerce",
        )
        out["industry_turnover_share_change_3d"] = live_flow_change
        live_strength = pd.to_numeric(
            out["live_industry_strength_percentile"], errors="coerce"
        ) * 100.0
        if "industry_relative_strength" not in out.columns:
            out["industry_relative_strength"] = live_strength
        else:
            out["industry_relative_strength"] = pd.to_numeric(
                out["industry_relative_strength"], errors="coerce"
            ).combine_first(live_strength)

    # Convert only explicitly observed lifecycle/policy states. UNKNOWN stays
    # missing; a missing catalyst archive is never interpreted as neutral or negative.
    if "live_mainline_phase_gated" in out:
        phase_scores = {
            "IGNITION": 100.0,
            "IGNITION_UNCONFIRMED": 55.0,
            "TRANSITION": 60.0,
            "PEAK": 40.0,
            "DECAY": 0.0,
        }
        out["live_mainline_phase_score"] = (
            out["live_mainline_phase_gated"].astype(str).map(phase_scores)
        )
    else:
        out["live_mainline_phase_score"] = np.nan
    if "live_startup_policy_status" in out:
        policy_scores = {
            "POSITIVE_CATALYST": 100.0,
            "NEGATIVE_CATALYST": 0.0,
        }
        out["live_policy_catalyst_score"] = (
            out["live_startup_policy_status"].astype(str).map(policy_scores)
        )
    else:
        out["live_policy_catalyst_score"] = np.nan

    ranked_columns = []
    available_factors = []
    for factor_name, (source_column, higher_is_better, _) in RESEARCH_FACTORS.items():
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
    weights = pd.Series(
        {name + "_percentile": weight for name, (_, _, weight) in RESEARCH_FACTORS.items()},
        dtype=float,
    ).reindex(ranked_columns)
    observed_weights = matrix.notna().mul(weights, axis=1)
    weighted_score = (
        matrix.fillna(0).mul(weights, axis=1).sum(axis=1)
        / observed_weights.sum(axis=1).replace(0, np.nan)
    )
    score_context_ready = mainline_anchor_ready | allow_quote_only_fallback
    out["research_opportunity_score"] = weighted_score.where(
        (out["research_factor_count"] >= min_observed_factors)
        & score_context_ready
    )
    out["research_opportunity_rank"] = (
        out["research_opportunity_score"].rank(method="min", ascending=False).astype("Int64")
    )
    # Separate genuine mainline-backed opportunities from momentum-only fallbacks.
    # An industry label or same-day relative strength alone is not proof of a live mainline.
    mainline_evidence_columns = [
        "industry_flow_concentration_percentile",
        "mainline_lifecycle_percentile",
        "policy_catalyst_percentile",
    ]
    mainline_evidence_count = out[mainline_evidence_columns].notna().sum(axis=1)
    out["research_mainline_evidence_count"] = mainline_evidence_count.astype(int)
    out["research_rank_tier"] = np.where(
        mainline_evidence_count.gt(0),
        "MAINLINE_BACKED_RESEARCH",
        "MOMENTUM_ONLY_WATCH",
    )
    out["research_rank_priority"] = np.where(mainline_evidence_count.gt(0), 0, 1)
    out["research_rank_status"] = np.select(
        [
            out["research_opportunity_score"].notna() & mainline_anchor_ready,
            out["research_opportunity_score"].notna() & ~mainline_anchor_ready,
            ~mainline_anchor_ready,
        ],
        [
            "RANKED_RESEARCH_ONLY",
            "RANKED_QUOTE_ONLY_NO_INDUSTRY",
            "MISSING_MAINLINE_CONTEXT",
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
        ["research_rank_priority", "research_opportunity_score", "research_factor_count", "symbol"],
        ascending=[True, False, False, True], na_position="last",
    ).reset_index(drop=True)
    audit = {
        "status": (
            "READY_RESEARCH_ONLY"
            if out["research_opportunity_score"].notna().any() and mainline_anchor_ready.any()
            else "READY_QUOTE_ONLY_FALLBACK"
            if out["research_opportunity_score"].notna().any()
            else "MAINLINE_CONTEXT_UNAVAILABLE"
            if not mainline_anchor_ready.any() and not allow_quote_only_fallback
            else "INSUFFICIENT_RESEARCH_FACTOR_COVERAGE"
        ),
        "row_count": int(len(out)),
        "ranked_count": int(out["research_opportunity_score"].notna().sum()),
        "unranked_count": int(out["research_opportunity_score"].isna().sum()),
        "mainline_context_ready_count": int(mainline_anchor_ready.sum()),
        "missing_mainline_context_count": int((~mainline_anchor_ready).sum()),
        "min_observed_factors": min_observed_factors,
        "factor_count": len(RESEARCH_FACTORS),
        "factor_sources": {name: source for name, (source, _, _) in RESEARCH_FACTORS.items()},
        "factor_weights": {name: weight for name, (_, _, weight) in RESEARCH_FACTORS.items() if weight > 0},
        "mainline_backed_ranked_count": int((out["research_rank_tier"] == "MAINLINE_BACKED_RESEARCH").sum()),
        "momentum_only_watch_count": int((out["research_rank_tier"] == "MOMENTUM_ONLY_WATCH").sum()),
        "rank_priority_policy": "mainline-backed opportunities precede momentum-only watches; no missing factor imputation",
        "score_method": "weighted_observed_percentiles_renormalized_per_row",
        "allow_quote_only_fallback": allow_quote_only_fallback,
        "uses_raw_cross_session_returns": False,
        "historical_pit_eligible": False,
        "formal_m5_score": False,
        "buy_authorization": False,
    }
    return out, audit


def build_final_candidate_shortlist(ranked: pd.DataFrame, *, max_candidates: int = 30, max_per_industry: int = 3, min_amount: float = 100_000_000, min_pct_change: float = -5.0, max_pct_change: float = 9.5) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Build a diversified research watchlist; never grants buy authorization."""
    if max_candidates < 1 or max_per_industry < 1:
        raise ValueError("shortlist limits must be positive")
    if ranked is None or ranked.empty:
        empty = pd.DataFrame()
        return empty, empty, {"status": "NO_RANKED_CANDIDATES", "candidate_count": 0, "event_watch_count": 0, "buy_authorization": False}
    data = ranked.copy()
    # Normalize both live quote schema and normalized research schema. The
    # production watchlist currently carries Chinese provider column names.
    aliases = {
        "price": ("最新价", "收盘", "close"),
        "amount": ("当日成交额", "成交额", "turnover"),
        "pct_chg": ("涨跌幅%", "涨跌幅", "pct_change"),
    }
    for target, candidates in aliases.items():
        if target not in data:
            source = next((column for column in candidates if column in data), None)
            data[target] = data[source] if source is not None else np.nan
        data[target] = pd.to_numeric(data[target], errors="coerce")
    if "research_opportunity_score" not in data:
        data["research_opportunity_score"] = np.nan
    data["research_opportunity_score"] = pd.to_numeric(
        data["research_opportunity_score"], errors="coerce"
    )
    if "symbol" not in data:
        if "代码" not in data: raise ValueError("ranked candidates require symbol or 代码")
        data["symbol"] = data["代码"].astype(str).str.strip().str.zfill(6)
    data["symbol"] = data["symbol"].astype(str).str.strip().str.zfill(6)
    industry_col = "live_industry_level1" if "live_industry_level1" in data else ("current_industry_level1" if "current_industry_level1" in data else None)
    data["shortlist_industry"] = data[industry_col].fillna("行业未知").astype(str) if industry_col else "行业未知"
    valid = data[data["research_opportunity_score"].notna() & data["price"].gt(0) & data["amount"].ge(min_amount)].copy()
    event_watch = valid[valid["pct_chg"].ge(max_pct_change)].copy()
    event_watch["event_watch_reason"] = "涨幅过大，单列强势事件观察，不进入普通候选池"
    ordinary = valid[valid["pct_chg"].ge(min_pct_change) & valid["pct_chg"].lt(max_pct_change)].copy()
    phase_column = next((
        column for column in ("live_mainline_phase_gated", "mainline_phase_gated")
        if column in ordinary.columns
    ), None)
    phase_filtered_count = 0
    if phase_column:
        phase_values = ordinary[phase_column].fillna("UNKNOWN").astype(str).str.upper()
        # The shortlist is a research watchlist, not an entry gate. Explicit
        # PEAK/DECAY phases are excluded; UNKNOWN lifecycle keeps discovery
        # alive but remains visibly non-authorizing until M3 confirms a phase.
        phase_eligible = ~phase_values.isin({"PEAK", "DECAY"})
        phase_filtered_count = int((~phase_eligible).sum())
        ordinary = ordinary.loc[phase_eligible].copy()
    if "research_rank_priority" in ordinary:
        ordinary["research_rank_priority"] = pd.to_numeric(ordinary["research_rank_priority"], errors="coerce").fillna(1)
        ordinary = ordinary.sort_values(["research_rank_priority", "research_opportunity_score", "amount", "symbol"], ascending=[True, False, False, True])
    else:
        ordinary = ordinary.sort_values(["research_opportunity_score", "amount", "symbol"], ascending=[False, False, True])
    # Market emotion controls the breadth of the research watchlist, not the
    # relative stock score. Unknown emotion preserves discovery but is explicit.
    regime_caps = {
        "ICE": 10,
        "RECOVERY": max_candidates,
        "TRANSITION": min(max_candidates, 20),
        "HEAT": min(max_candidates, 15),
        "EXHAUSTION": min(max_candidates, 5),
    }
    regime_state = (
        str(data["market_cycle_state"].dropna().iloc[0])
        if "market_cycle_state" in data and data["market_cycle_state"].notna().any()
        else "UNKNOWN"
    )
    regime_cap = int(regime_caps.get(regime_state, max_candidates))
    effective_candidate_cap = min(max_candidates, regime_cap)
    selected = []; industry_counts = {}
    for _, row in ordinary.iterrows():
        industry = row["shortlist_industry"]
        if industry in {"行业未知", "nan", "None", ""}:
            industry = "UNKNOWN:" + str(row["symbol"])
        if industry_counts.get(industry, 0) >= max_per_industry: continue
        selected.append(row); industry_counts[industry] = industry_counts.get(industry, 0) + 1
        if len(selected) >= effective_candidate_cap: break
    shortlist = pd.DataFrame(selected, columns=ordinary.columns).reset_index(drop=True)
    if not shortlist.empty:
        shortlist.insert(0, "final_candidate_rank", np.arange(1, len(shortlist) + 1))
        shortlist["candidate_status"] = np.where(
            shortlist.get("research_rank_tier", pd.Series("MOMENTUM_ONLY_WATCH", index=shortlist.index)).eq("MAINLINE_BACKED_RESEARCH"),
            "MAINLINE_BACKED_RESEARCH_WATCH",
            "MOMENTUM_ONLY_WATCH",
        )
        shortlist["entry_confirmation_status"] = "PENDING_15M_CONFIRMATION"
        shortlist["buy_authorization"] = False
        if phase_column:
            shortlist_phase = shortlist[phase_column].fillna("UNKNOWN").astype(str).str.upper()
            shortlist["mainline_lifecycle_status"] = np.where(
                shortlist_phase.eq("UNKNOWN"), "UNKNOWN_RESEARCH_ONLY", "OBSERVED"
            )
        else:
            shortlist["mainline_lifecycle_status"] = "UNKNOWN_RESEARCH_ONLY"
        shortlist["risk_note"] = np.select(
            [shortlist["pct_chg"].ge(7.5), shortlist["pct_chg"].le(-3.0)],
            ["高涨幅，严禁按排名追价", "当日偏弱，等待结构转强"],
            default="等待15分钟结构与风险确认",
        )
        unknown_lifecycle = shortlist["mainline_lifecycle_status"].eq("UNKNOWN_RESEARCH_ONLY")
        shortlist.loc[unknown_lifecycle, "risk_note"] = (
            shortlist.loc[unknown_lifecycle, "risk_note"].astype(str)
            + "；行业主线周期未知，仅作研究观察，不构成买入许可"
        )
        if "market_cycle_state" in shortlist.columns:
            market_risk = shortlist["market_cycle_state"].astype(str).map({
                "ICE": "市场情绪冰点，缩小观察范围，等待修复确认",
                "RECOVERY": "市场情绪修复，可关注主线持续性",
                "TRANSITION": "市场情绪过渡期，优先观察主线确认",
                "HEAT": "市场情绪高潮，缩小观察范围，避免追高",
                "EXHAUSTION": "市场情绪退潮，严格限制新增进攻候选",
                "UNKNOWN": "市场情绪数据未知，不授予进攻型信号",
            })
            has_market_risk = market_risk.notna()
            shortlist.loc[has_market_risk, "risk_note"] = (
                shortlist.loc[has_market_risk, "risk_note"].astype(str)
                + "；" + market_risk.loc[has_market_risk]
            )
    if not event_watch.empty:
        event_watch = event_watch.sort_values(["research_opportunity_score", "amount"], ascending=[False, False]).reset_index(drop=True)
        event_watch.insert(0, "event_watch_rank", np.arange(1, len(event_watch) + 1))
        event_watch["buy_authorization"] = False
    audit = {"status": "READY_RESEARCH_WATCH" if not shortlist.empty else "NO_ORDINARY_CANDIDATES", "ranked_input_count": int(len(data)), "liquid_valid_count": int(len(valid)), "mainline_phase_filtered_count": phase_filtered_count, "mainline_phase_filter": "exclude observed PEAK/DECAY; retain UNKNOWN as non-authorizing research watch", "candidate_count": int(len(shortlist)), "event_watch_count": int(len(event_watch)), "market_cycle_state": regime_state, "market_regime_candidate_cap": effective_candidate_cap, "market_regime_policy": "ICE<=10; RECOVERY<=max; TRANSITION<=20; HEAT<=15; EXHAUSTION<=5; UNKNOWN<=max_with_warning", "industry_cap": int(max_per_industry), "minimum_amount": float(min_amount), "ordinary_pct_change_range": [float(min_pct_change), float(max_pct_change)], "entry_confirmation": "PENDING_15M_CONFIRMATION", "buy_authorization": False}
    return shortlist, event_watch, audit
