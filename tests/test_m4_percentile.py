import numpy as np
import pandas as pd
import pytest

from src.model.m4_percentile import audited_percentile_rank


def pool(values, date="2026-11-10"):
    return pd.DataFrame({
        "trade_date": [date] * len(values),
        "symbol": [f"{i:06d}" for i in range(len(values))],
        "raw": values,
    })


def test_audited_rank_records_scope_and_complete_coverage():
    frame = pool([1.0, 2.0, 2.0, 4.0])
    ranks, audit = audited_percentile_rank(frame, "raw", scope_id="top_350_turnover_universe", expected_universe_size=4)
    assert ranks.tolist() == [25.0, 62.5, 62.5, 100.0]
    assert audit.iloc[0].scope_id == "top_350_turnover_universe"
    assert audit.iloc[0].status == "ranked"
    assert audit.iloc[0].factor_coverage == 1.0


def test_incomplete_universe_fails_closed():
    frame = pool([1.0, 2.0, 3.0])
    ranks, audit = audited_percentile_rank(frame, "raw", scope_id="declared_pool", expected_universe_size=4)
    assert ranks.isna().all()
    assert audit.iloc[0].status == "universe_size_mismatch"


def test_low_factor_coverage_fails_closed_and_preserves_missing():
    frame = pool([1.0, 2.0, np.nan, np.nan])
    ranks, audit = audited_percentile_rank(frame, "raw", scope_id="declared_pool", expected_universe_size=4, min_coverage=0.75)
    assert ranks.isna().all()
    assert audit.iloc[0].status == "insufficient_factor_coverage"
    assert audit.iloc[0].finite_factor_count == 2


def test_lower_is_better_reverses_percentile_direction():
    frame = pool([1.0, 2.0, 3.0])
    ranks, _ = audited_percentile_rank(frame, "raw", scope_id="declared_pool", expected_universe_size=3, higher_is_better=False)
    assert ranks.tolist() == pytest.approx([100.0, 66.6666666667, 33.3333333333])


def test_duplicate_symbol_in_date_is_rejected():
    frame = pool([1.0, 2.0])
    frame.loc[1, "symbol"] = frame.loc[0, "symbol"]
    with pytest.raises(ValueError, match="duplicate"):
        audited_percentile_rank(frame, "raw", scope_id="declared_pool", expected_universe_size=2)


def test_scope_and_coverage_arguments_are_mandatory_and_validated():
    frame = pool([1.0])
    with pytest.raises(ValueError, match="scope_id"):
        audited_percentile_rank(frame, "raw", scope_id=" ", expected_universe_size=1)
    with pytest.raises(ValueError, match="min_coverage"):
        audited_percentile_rank(frame, "raw", scope_id="pool", expected_universe_size=1, min_coverage=0)
