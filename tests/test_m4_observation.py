import numpy as np
import pandas as pd

from src.model.m4_observation import rank_price_observations


def sample(values):
    n = len(values)
    return pd.DataFrame({
        "trade_date": ["2026-11-10"] * n,
        "symbol": [f"{i:06d}" for i in range(n)],
        "CLV": values,
        "BodyStrength": values,
        "return_5": values,
        "return_10": values,
        "return_20": values,
        "breakout20": values,
        "upper_shadow_ratio": values,
    })


def test_price_observation_ranks_are_audited_and_directional():
    frame = sample([0.2, 0.5, 0.8, 1.0])
    ranked, audit = rank_price_observations(frame, expected_universe_size=4)
    assert ranked["CLV_pct"].tolist() == [25.0, 50.0, 75.0, 100.0]
    assert ranked["upper_shadow_ratio_pct"].tolist() == [100.0, 75.0, 50.0, 25.0]
    assert set(audit["status"]) == {"ranked"}
    assert set(audit["scope_id"]) == {"top_350_turnover_snapshot"}


def test_incomplete_peer_universe_keeps_every_rank_missing():
    frame = sample([0.2, 0.5, 0.8])
    ranked, audit = rank_price_observations(frame, expected_universe_size=4)
    assert ranked.filter(regex="_pct$").isna().all().all()
    assert set(audit["status"]) == {"universe_size_mismatch"}


def test_missing_values_fail_closed_when_coverage_below_floor():
    frame = sample([0.2, np.nan, np.nan, 1.0])
    ranked, audit = rank_price_observations(
        frame, expected_universe_size=4, min_coverage=0.75
    )
    assert ranked.filter(regex="_pct$").isna().all().all()
    assert set(audit["status"]) == {"insufficient_factor_coverage"}


def test_conditional_volume_and_pullback_fields_are_not_arbitrarily_ranked():
    frame = sample([0.2, 0.5, 0.8])
    frame["Pullback"] = [-0.01, -0.03, -0.05]
    frame["amount_ma5_to_ma20"] = [0.8, 1.2, 2.0]
    ranked, _ = rank_price_observations(frame, expected_universe_size=3)
    assert "Pullback_pct" not in ranked
    assert "amount_ma5_to_ma20_pct" not in ranked
