import pandas as pd
import pytest

from src.model.r21_core import A_WEIGHTS, B_WEIGHTS, C_WEIGHTS
from src.model.pipeline import evaluate_candidates, evaluate_row


def make_row(state="RISK_ON"):
    row = {"symbol": "000001", "market_state": state}
    for weights in (A_WEIGHTS, B_WEIGHTS, C_WEIGHTS):
        row.update({key: 90.0 for key in weights})
    row.update({
        "industry_top_30pct": True,
        "stock_residual_top_20pct": True,
        "amount_ma20_ge_100m": True,
        "breakout_or_confirmed_retest": True,
        "no_high_volume_stall_or_repeated_rejection": True,
        "event_subtype": "close_up_sealed",
        "event_data_verified": True,
        "liquidity_and_risk_pass": True,
        "announcement_risk_pass": True,
        "pre_pullback_residual_positive": True,
        "pullback_between_0_5_and_4pct": True,
        "pullback_volume_le_80pct_prior5d": True,
        "stop_or_reversal_confirmed": True,
        "industry_not_weak": True,
        "no_material_negative_event": True,
    })
    return row


def by_strategy(results):
    return {item["strategy"]: item for item in results}


def test_m5_adapter_evaluates_a_and_c_but_defers_b_until_next_day():
    result = by_strategy(evaluate_row(make_row(), stage="close"))
    assert result["A"]["eligible"] and result["A"]["score"] == pytest.approx(90)
    assert result["C"]["eligible"] and result["C"]["score"] == pytest.approx(90)
    assert not result["B"]["eligible"]
    assert result["B"]["score"] is None


def test_m5_b_requires_next_day_factor_and_does_not_infer_it():
    row = make_row()
    result = by_strategy(evaluate_row(row, stage="next_day"))
    assert not result["B"]["eligible"]
    assert result["B"]["score"] is None
    row["next_day_price_action"] = 90.0
    result = by_strategy(evaluate_row(row, stage="next_day"))
    assert result["B"]["eligible"]
    assert result["B"]["score"] == pytest.approx(90)


def test_m5_neutral_market_raises_floor_to_85():
    row = make_row("NEUTRAL")
    for key in A_WEIGHTS:
        row[key] = 84.99
    result = by_strategy(evaluate_row(row))
    assert not result["A"]["eligible"]
    assert result["A"]["minimum_score"] == 85.0
    assert "below_minimum_score:85" in result["A"]["reasons"]


def test_m5_risk_off_rejects_even_high_factor_scores():
    result = by_strategy(evaluate_row(make_row("RISK_OFF")))
    assert all(not result[s]["eligible"] for s in ("A", "B", "C"))
    assert all("market_risk_off" in result[s]["reasons"] for s in ("A", "B", "C"))


def test_m5_missing_factor_and_missing_gate_fail_closed():
    row = make_row()
    del row["catalyst"]
    row["industry_top_30pct"] = None
    result = by_strategy(evaluate_row(row))
    assert result["A"]["score"] is None
    assert not result["A"]["eligible"]
    assert "missing_or_invalid_factor:catalyst" in result["A"]["reasons"]


def test_m5_batch_keeps_rejected_rows_and_emits_three_strategy_records():
    frame = pd.DataFrame([make_row(), {**make_row(), "symbol": "000002", "market_state": "UNKNOWN"}])
    out = evaluate_candidates(frame)
    assert len(out) == 6
    assert set(out.symbol) == {"000001", "000002"}
    assert not out.loc[out.symbol == "000002", "eligible"].any()
