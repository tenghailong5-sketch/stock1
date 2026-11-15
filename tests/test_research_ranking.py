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
