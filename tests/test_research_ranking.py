import pandas as pd

from src.model.research_ranking import build_research_opportunity_ranking, build_final_candidate_shortlist


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
        "industry_strength": 0.20,
        "current_momentum": 0.15,
        "current_liquidity": 0.10,
        "industry_flow_concentration": 0.20,
        "mainline_lifecycle": 0.15,
        "policy_catalyst": 0.10,
        "emotion_ice_reversal": 0.10,
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


def test_sustained_industry_flow_concentration_is_used_when_observed():
    watchlist = pd.DataFrame({
        "代码": ["000001", "000002"],
        "涨跌幅%": [3.0, 3.0],
        "当日成交额": [200e6, 200e6],
    })
    m5_input = pd.DataFrame({
        "symbol": ["000001", "000002"],
        "industry_relative_strength": [80.0, 80.0],
    })
    industry = pd.DataFrame({
        "symbol": ["000001", "000002"],
        "current_industry_level1": ["计算机", "传媒"],
        "strength_percentile": [0.8, 0.8],
        "turnover_share_change_3d": [0.08, -0.08],
    })
    ranked, audit = build_research_opportunity_ranking(
        watchlist, m5_input, industry_context=industry,
    )
    by_symbol = ranked.set_index("symbol")
    assert by_symbol.loc["000001", "industry_flow_concentration_percentile"] > by_symbol.loc["000002", "industry_flow_concentration_percentile"]
    assert by_symbol.loc["000001", "research_opportunity_score"] > by_symbol.loc["000002", "research_opportunity_score"]
    assert audit["factor_weights"]["industry_flow_concentration"] == 0.15


def test_gated_mainline_phase_and_policy_catalyst_adjust_research_score():
    watchlist = pd.DataFrame({
        "代码": ["000001", "000002"],
        "涨跌幅%": [3.0, 3.0],
        "当日成交额": [200e6, 200e6],
    })
    m5_input = pd.DataFrame({
        "symbol": ["000001", "000002"],
        "industry_relative_strength": [80.0, 80.0],
    })
    industry = pd.DataFrame({
        "symbol": ["000001", "000002"],
        "current_industry_level1": ["计算机", "传媒"],
        "strength_percentile": [0.8, 0.8],
        "mainline_phase_gated": ["IGNITION", "DECAY"],
        "startup_policy_status": ["POSITIVE_CATALYST", "NEGATIVE_CATALYST"],
    })
    ranked, audit = build_research_opportunity_ranking(
        watchlist, m5_input, industry_context=industry,
    )
    by_symbol = ranked.set_index("symbol")
    assert by_symbol.loc["000001", "live_mainline_phase_score"] == 100.0
    assert by_symbol.loc["000002", "live_mainline_phase_score"] == 0.0
    assert by_symbol.loc["000001", "live_policy_catalyst_score"] == 100.0
    assert by_symbol.loc["000002", "live_policy_catalyst_score"] == 0.0
    assert by_symbol.loc["000001", "research_opportunity_score"] > by_symbol.loc["000002", "research_opportunity_score"]
    assert audit["factor_weights"]["mainline_lifecycle"] == 0.10
    assert audit["factor_weights"]["policy_catalyst"] == 0.05


def test_market_emotion_limits_research_watchlist_by_regime():
    rows = []
    for state in ["ICE", "RECOVERY", "TRANSITION", "HEAT", "EXHAUSTION", "UNKNOWN"]:
        for i in range(30):
            rows.append({
                "symbol": f"{len(rows)+1:06d}",
                "最新价": 10.0,
                "涨跌幅%": 3.0,
                "当日成交额": 200_000_000,
                "research_opportunity_score": 100.0 - len(rows) * 0.01,
                "current_industry_level1": f"行业{state}-{i}",
                "market_cycle_state": state,
            })
    ranked = pd.DataFrame(rows)
    expected = {
        "ICE": 10, "RECOVERY": 30, "TRANSITION": 20,
        "HEAT": 15, "EXHAUSTION": 5, "UNKNOWN": 30,
    }
    for state, count in expected.items():
        part = ranked[ranked["market_cycle_state"].eq(state)].copy()
        shortlist, _, audit = build_final_candidate_shortlist(
            part, max_candidates=30, max_per_industry=3,
        )
        assert len(shortlist) == count
        assert audit["market_cycle_state"] == state
        assert audit["market_regime_candidate_cap"] == count
        assert not shortlist["buy_authorization"].any()



def test_final_shortlist_excludes_decay_and_peak_industries():
    ranked = pd.DataFrame({
        "symbol": ["000001", "000002", "000003", "000004"],
        "最新价": [10.0, 10.0, 10.0, 10.0],
        "涨跌幅%": [3.0, 3.0, 3.0, 3.0],
        "当日成交额": [300e6, 300e6, 300e6, 300e6],
        "research_opportunity_score": [99.0, 98.0, 97.0, 96.0],
        "live_industry_level1": ["电子", "计算机", "银行", "通信"],
        "live_mainline_phase_gated": [
            "IGNITION_UNCONFIRMED", "TRANSITION", "DECAY", "PEAK"
        ],
    })
    shortlist, _, audit = build_final_candidate_shortlist(
        ranked, max_candidates=10, max_per_industry=3,
    )
    assert shortlist["symbol"].tolist() == ["000001", "000002"]
    assert audit["mainline_phase_filtered_count"] == 2
    assert not shortlist["buy_authorization"].any()


def test_event_price_in_diagnostics_are_carried_without_changing_rank_score():
    watchlist = pd.DataFrame({
        "代码": ["000001", "000002"],
        "涨跌幅%": [3.0, 3.0],
        "当日成交额": [200e6, 200e6],
    })
    m5_input = pd.DataFrame({
        "symbol": ["000001", "000002"],
        "industry_relative_strength": [80.0, 80.0],
    })
    base_context = pd.DataFrame({
        "symbol": ["000001", "000002"],
        "current_industry_level1": ["计算机", "传媒"],
        "strength_percentile": [0.8, 0.8],
    })
    price_in_context = base_context.assign(
        live_catalyst_candidate_count=[3, 1],
        live_event_price_in_state=["FULLY_PRICED", "UNPRICED"],
        live_event_price_in_reason=["runup_without_repricing", "no_material_reaction"],
        live_event_pre_discovery_excess_20d=[0.12, 0.01],
        live_event_post_discovery_excess_3d=[0.0, 0.025],
    )
    baseline, _ = build_research_opportunity_ranking(
        watchlist, m5_input, industry_context=base_context,
        min_observed_factors=2,
    )
    ranked, _ = build_research_opportunity_ranking(
        watchlist, m5_input, industry_context=price_in_context,
        min_observed_factors=2,
    )
    by_symbol = ranked.set_index("symbol")
    baseline_scores = baseline.set_index("symbol")["research_opportunity_score"]
    assert by_symbol.loc["000001", "live_event_price_in_state"] == "FULLY_PRICED"
    assert by_symbol.loc["000002", "live_event_price_in_state"] == "UNPRICED"
    assert by_symbol.loc["000001", "live_catalyst_candidate_count"] == 3
    assert by_symbol["research_opportunity_score"].equals(baseline_scores)


def test_unknown_mainline_phase_remains_in_research_watchlist_without_buy_authorization():
    ranked = pd.DataFrame({
        "symbol": ["000001", "000002", "000003"],
        "最新价": [10.0, 20.0, 30.0],
        "涨跌幅%": [2.0, 3.0, 4.0],
        "当日成交额": [300e6, 250e6, 200e6],
        "research_opportunity_score": [95.0, 90.0, 85.0],
        "live_industry_level1": ["电子", "计算机", "机械设备"],
        "live_mainline_phase_gated": ["UNKNOWN", None, "UNKNOWN"],
        "market_cycle_state": ["TRANSITION"] * 3,
    })
    shortlist, _, audit = build_final_candidate_shortlist(ranked, max_candidates=10)
    assert shortlist["symbol"].tolist() == ["000001", "000002", "000003"]
    assert shortlist["mainline_lifecycle_status"].eq("UNKNOWN_RESEARCH_ONLY").all()
    assert not shortlist["buy_authorization"].any()
    assert audit["mainline_phase_filtered_count"] == 0


def test_mainline_backed_candidates_rank_ahead_of_momentum_only_watch():
    watchlist = pd.DataFrame({
        "代码": ["000001", "000002", "000003"],
        "名称": ["momentum", "mainline", "mixed"],
        "最新价": [10, 10, 10],
        "涨跌幅%": [9.0, 2.0, 5.0],
        "当日成交额": [900e6, 200e6, 500e6],
    })
    m5 = pd.DataFrame({
        "symbol": ["000001", "000002", "000003"],
        "industry_relative_strength": [99, 80, 90],
    })
    industry = pd.DataFrame({
        "symbol": ["000001", "000002", "000003"],
        "current_industry_level1": ["银行", "计算机", "传媒"],
        "strength_percentile": [0.99, 0.80, 0.90],
        "turnover_share_change_3d": [None, 0.08, None],
        "mainline_phase_gated": [None, "IGNITION_UNCONFIRMED", None],
    })
    ranked, audit = build_research_opportunity_ranking(
        watchlist, m5, industry_context=industry,
    )
    assert ranked.iloc[0]["symbol"] == "000002"
    assert ranked.iloc[0]["research_rank_tier"] == "MAINLINE_BACKED_RESEARCH"
    assert ranked.iloc[1]["research_rank_tier"] == "MOMENTUM_ONLY_WATCH"
    assert audit["mainline_backed_ranked_count"] == 1
    assert audit["momentum_only_watch_count"] == 2
    shortlist, _, _ = build_final_candidate_shortlist(
        ranked, max_candidates=3, min_amount=100e6,
    )
    assert shortlist.iloc[0]["symbol"] == "000002"
    assert shortlist.iloc[1]["candidate_status"] == "MOMENTUM_ONLY_WATCH"


def test_research_ranking_rewards_recent_policy_and_ice_reversal_evidence():
    watchlist = pd.DataFrame({
        "代码": ["000001", "000002"],
        "涨跌幅%": [3.0, 3.0],
        "当日成交额": [200e6, 200e6],
    })
    m5_input = pd.DataFrame({
        "symbol": ["000001", "000002"],
        "industry_relative_strength": [80.0, 80.0],
    })
    industry = pd.DataFrame({
        "symbol": ["000001", "000002"],
        "current_industry_level1": ["计算机", "传媒"],
        "strength_percentile": [0.8, 0.8],
        "mainline_phase_gated": ["IGNITION", "IGNITION"],
        "startup_policy_status": ["POSITIVE_CATALYST", "POSITIVE_CATALYST"],
        "policy_catalyst_recency_score": [100.0, 40.0],
        "emotion_reversal_component": [True, False],
        "turnover_share_change_3d": [0.08, -0.08],
    })
    ranked, audit = build_research_opportunity_ranking(
        watchlist, m5_input, industry_context=industry,
    )
    by_symbol = ranked.set_index("symbol")
    assert by_symbol.loc["000001", "live_emotion_reversal_score"] == 100.0
    assert by_symbol.loc["000002", "live_emotion_reversal_score"] == 0.0
    assert by_symbol.loc["000001", "live_policy_catalyst_recency_score"] == 100.0
    assert by_symbol.loc["000002", "live_policy_catalyst_recency_score"] == 40.0
    assert by_symbol.loc["000001", "research_opportunity_score"] > by_symbol.loc["000002", "research_opportunity_score"]
    assert audit["factor_weights"]["emotion_ice_reversal"] == 0.10
