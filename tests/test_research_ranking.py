import pandas as pd

from src.model.research_ranking import build_research_opportunity_ranking


def test_research_ranking_uses_only_observed_same_session_and_industry_factors():
    watchlist = pd.DataFrame({
        "代码": ["000001", "000002", "000003"],
        "名称": ["A", "B", "C"],
        "CLV": [0.9, 0.5, 0.2],
        "BodyStrength": [0.8, 0.1, -0.2],
        "upper_shadow_ratio": [0.05, 0.2, 0.4],
        "20日平均成交额": [300_000_000, 200_000_000, 100_000_000],
        "5日收益%": [10.0, -5.0, 2.0],
        "20日收益%": [30.0, -10.0, 1.0],
    })
    m5_input = pd.DataFrame({
        "symbol": ["000001", "000002", "000003"],
        "industry_relative_strength": [95.0, 60.0, 20.0],
        "price_structure": [90.0, 50.0, 10.0],
        "trade_quality": [100.0, 66.0, 33.0],
    })
    ranked, audit = build_research_opportunity_ranking(watchlist, m5_input)
    assert ranked.iloc[0]["symbol"] == "000001"
    assert audit["ranked_count"] == 3
    assert not audit["uses_raw_cross_session_returns"]
    assert ranked["research_rank_formal_m5_score"].eq(False).all()
    assert ranked["research_rank_buy_authorization"].eq(False).all()
    assert "快速5日收益%" not in audit["factor_sources"].values()


def test_research_ranking_keeps_missing_factors_missing_and_requires_three():
    watchlist = pd.DataFrame({
        "代码": ["000001", "000002"],
        "CLV": [0.9, None],
        "BodyStrength": [0.5, None],
        "upper_shadow_ratio": [0.1, None],
        "20日平均成交额": [100_000_000, None],
    })
    m5_input = pd.DataFrame({
        "symbol": ["000001", "000002"],
        "industry_relative_strength": [80.0, None],
        "price_structure": [90.0, None],
        "trade_quality": [100.0, None],
    })
    ranked, audit = build_research_opportunity_ranking(watchlist, m5_input)
    assert ranked.loc[ranked.symbol.eq("000001"), "research_opportunity_score"].notna().all()
    missing = ranked.loc[ranked.symbol.eq("000002")].iloc[0]
    assert pd.isna(missing["research_opportunity_score"])
    assert missing["research_factor_count"] == 0
    assert audit["unranked_count"] == 1


def test_research_ranking_rejects_duplicate_symbols():
    watchlist = pd.DataFrame({"代码": ["000001", "000001"]})
    m5_input = pd.DataFrame({"symbol": ["000001"]})
    try:
        build_research_opportunity_ranking(watchlist, m5_input)
    except ValueError as exc:
        assert "duplicate symbol" in str(exc)
    else:
        raise AssertionError("duplicate symbols must be rejected")


def test_research_ranking_uses_forward_mainline_context_without_formal_m5_promotion():
    watchlist = pd.DataFrame({
        "代码": ["000001", "000002"],
        "BodyStrength": [0.8, 0.2],
        "upper_shadow_ratio": [0.05, 0.3],
    })
    m5_input = pd.DataFrame({
        "symbol": ["000001", "000002"],
        "price_structure": [80.0, 20.0],
        "trade_quality": [90.0, 30.0],
    })
    industry_context = pd.DataFrame({
        "symbol": ["000001", "000002"],
        "industry_code": ["801780", "801180"],
        "current_industry_level1": ["银行", "房地产"],
        "mainline_phase": ["IGNITION", "DECAY"],
        "mainline_phase_gated": ["IGNITION_UNCONFIRMED", "DECAY"],
        "strength_percentile": [0.9, 0.2],
        "startup_evidence_status": ["UNKNOWN", "UNKNOWN"],
    })
    ranked, audit = build_research_opportunity_ranking(
        watchlist, m5_input, industry_context=industry_context,
    )
    leader = ranked.set_index("symbol").loc["000001"]
    assert leader["industry_relative_strength"] == 90.0
    assert leader["live_mainline_phase_gated"] == "IGNITION_UNCONFIRMED"
    assert audit["ranked_count"] == 2
    assert ranked["research_rank_formal_m5_score"].eq(False).all()
    assert ranked["research_rank_buy_authorization"].eq(False).all()



def test_research_ranking_does_not_rank_without_mainline_anchor():
    watchlist = pd.DataFrame({
        "代码": ["000001", "000002"],
        "BodyStrength": [0.9, 0.8],
        "upper_shadow_ratio": [0.05, 0.08],
    })
    m5_input = pd.DataFrame({
        "symbol": ["000001", "000002"],
        "price_structure": [95.0, 90.0],
        "trade_quality": [98.0, 92.0],
    })
    ranked, audit = build_research_opportunity_ranking(watchlist, m5_input)
    assert audit["status"] == "MAINLINE_CONTEXT_UNAVAILABLE"
    assert audit["ranked_count"] == 0
    assert audit["missing_mainline_context_count"] == 2
    assert ranked.research_rank_status.eq("MISSING_MAINLINE_CONTEXT").all()
    assert ranked.research_opportunity_score.isna().all()



def test_quote_only_current_snapshot_can_rank_with_live_industry_anchor():
    watchlist = pd.DataFrame({
        "代码": ["000001", "000002", "000003"],
        "名称": ["A", "B", "C"],
        "涨跌幅%": [5.0, 2.0, -1.0],
        "当日成交额": [300_000_000, 200_000_000, 100_000_000],
    })
    m5_input = pd.DataFrame({
        "symbol": ["000001", "000002", "000003"],
        "industry_relative_strength": [95.0, 60.0, 20.0],
    })
    ranked, audit = build_research_opportunity_ranking(watchlist, m5_input)
    assert audit["ranked_count"] == 3
    assert ranked.iloc[0]["symbol"] == "000001"
    assert ranked["research_rank_status"].eq("RANKED_RESEARCH_ONLY").all()
    assert ranked["research_rank_formal_m5_score"].eq(False).all()
    assert ranked["research_rank_buy_authorization"].eq(False).all()



def test_weighted_quote_only_score_and_final_shortlist_are_diversified():
    from src.model.research_ranking import build_final_candidate_shortlist

    watchlist = pd.DataFrame({
        "代码": ["000001", "000002", "000003", "000004", "000005", "000006"],
        "名称": ["A", "B", "C", "D", "E", "F"],
        "最新价": [10, 11, 12, 13, 14, 15],
        "涨跌幅%": [4.0, 3.0, 2.0, 5.0, 4.5, 12.0],
        "当日成交额": [300e6, 250e6, 200e6, 180e6, 160e6, 400e6],
    })
    m5_input = pd.DataFrame({
        "symbol": ["000001", "000002", "000003", "000004", "000005", "000006"],
        "industry_relative_strength": [95, 90, 85, 80, 75, 99],
    })
    industry = pd.DataFrame({
        "symbol": ["000001", "000002", "000003", "000004", "000005", "000006"],
        "current_industry_level1": ["银行", "银行", "银行", "计算机", "计算机", "传媒"],
        "strength_percentile": [0.95, 0.90, 0.85, 0.80, 0.75, 0.99],
    })
    ranked, audit = build_research_opportunity_ranking(
        watchlist, m5_input, industry_context=industry
    )
    shortlist, event_watch, shortlist_audit = build_final_candidate_shortlist(
        ranked, max_candidates=5, max_per_industry=2, min_amount=100e6
    )
    assert audit["score_method"] == "weighted_observed_percentiles_renormalized_per_row"
    assert audit["factor_weights"] == {
        "industry_strength": 0.35,
        "current_momentum": 0.45,
        "current_liquidity": 0.20,
    }
    assert len(shortlist) == 4
    assert shortlist["shortlist_industry"].value_counts().max() <= 2
    assert len(event_watch) == 1
    assert event_watch.iloc[0]["symbol"] == "000006"
    assert shortlist["entry_confirmation_status"].eq("PENDING_15M_CONFIRMATION").all()
    assert shortlist["buy_authorization"].eq(False).all()
    assert shortlist_audit["buy_authorization"] is False



def test_quote_only_fallback_ranks_when_industry_context_is_unavailable():
    watchlist = pd.DataFrame({
        "代码": ["000001", "000002", "000003"],
        "最新价": [10.0, 11.0, 12.0],
        "涨跌幅%": [5.0, 2.0, -1.0],
        "当日成交额": [300e6, 200e6, 100e6],
    })
    m5_input = pd.DataFrame({"symbol": ["000001", "000002", "000003"]})
    ranked, audit = build_research_opportunity_ranking(
        watchlist, m5_input, min_observed_factors=2,
        allow_quote_only_fallback=True,
    )
    assert audit["ranked_count"] == 3
    assert audit["status"] == "READY_QUOTE_ONLY_FALLBACK"
    assert audit["mainline_context_ready_count"] == 0
    assert ranked.research_rank_status.eq("RANKED_QUOTE_ONLY_NO_INDUSTRY").all()
    shortlist, _, shortlist_audit = build_final_candidate_shortlist(ranked, max_candidates=3)
    assert len(shortlist) == 3
    assert shortlist_audit["candidate_count"] == 3
    assert shortlist.buy_authorization.eq(False).all()
