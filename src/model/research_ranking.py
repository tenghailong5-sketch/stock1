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
    "industry_strength": ("industry_relative_strength", True, 0.35),
    "current_momentum": ("涨跌幅%", True, 0.45),
    "current_liquidity": ("当日成交额", True, 0.20),
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
    out["research_rank_status"] = np.select(
        [
            ~mainline_anchor_ready,
            out["research_opportunity_score"].notna(),
        ],
        [
            "MISSING_MAINLINE_CONTEXT",
            np.where(mainline_anchor_ready, "RANKED_RESEARCH_ONLY", "RANKED_QUOTE_ONLY_NO_INDUSTRY"),
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
    for col in ("price", "amount", "pct_chg", "research_opportunity_score"):
        if col not in data: data[col] = np.nan
        data[col] = pd.to_numeric(data[col], errors="coerce")
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
    ordinary = ordinary.sort_values(["research_opportunity_score", "amount", "symbol"], ascending=[False, False, True])
    selected = []; industry_counts = {}
    for _, row in ordinary.iterrows():
        industry = row["shortlist_industry"]
        if industry in {"行业未知", "nan", "None", ""}:
            industry = "UNKNOWN:" + str(row["symbol"])
        if industry_counts.get(industry, 0) >= max_per_industry: continue
        selected.append(row); industry_counts[industry] = industry_counts.get(industry, 0) + 1
        if len(selected) >= max_candidates: break
    shortlist = pd.DataFrame(selected, columns=ordinary.columns).reset_index(drop=True)
    if not shortlist.empty:
        shortlist.insert(0, "final_candidate_rank", np.arange(1, len(shortlist) + 1))
        shortlist["candidate_status"] = "RESEARCH_WATCH"
        shortlist["entry_confirmation_status"] = "PENDING_15M_CONFIRMATION"
        shortlist["buy_authorization"] = False
        shortlist["risk_note"] = np.select([shortlist["pct_chg"].ge(7.5), shortlist["pct_chg"].le(-3.0)], ["高涨幅，严禁按排名追价", "当日偏弱，等待结构转强"], default="等待15分钟结构与风险确认")
    if not event_watch.empty:
        event_watch = event_watch.sort_values(["research_opportunity_score", "amount"], ascending=[False, False]).reset_index(drop=True)
        event_watch.insert(0, "event_watch_rank", np.arange(1, len(event_watch) + 1))
        event_watch["buy_authorization"] = False
    audit = {"status": "READY_RESEARCH_WATCH" if not shortlist.empty else "NO_ORDINARY_CANDIDATES", "ranked_input_count": int(len(data)), "liquid_valid_count": int(len(valid)), "candidate_count": int(len(shortlist)), "event_watch_count": int(len(event_watch)), "industry_cap": int(max_per_industry), "minimum_amount": float(min_amount), "ordinary_pct_change_range": [float(min_pct_change), float(max_pct_change)], "entry_confirmation": "PENDING_15M_CONFIRMATION", "buy_authorization": False}
    return shortlist, event_watch, audit
