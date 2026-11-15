import pandas as pd

from src.fhm100 import select_candidate_history_topup_symbols


def test_candidate_history_topup_selects_highest_ranked_missing_history_only():
    frame = pd.DataFrame([
        {"symbol": "000003", "history_data_available": False, "research_opportunity_rank": 3},
        {"symbol": "000001", "history_data_available": False, "research_opportunity_rank": 1},
        {"symbol": "000002", "history_data_available": True, "research_opportunity_rank": 2},
        {"symbol": "000004", "history_data_available": False, "research_opportunity_rank": 4},
        {"symbol": "000001", "history_data_available": False, "research_opportunity_rank": 5},
    ])
    assert select_candidate_history_topup_symbols(frame, max_symbols=2) == [
        "000001", "000003"
    ]


def test_candidate_history_topup_respects_zero_and_negative_limits():
    frame = pd.DataFrame([
        {"symbol": "000001", "history_data_available": False, "research_opportunity_rank": 1},
    ])
    assert select_candidate_history_topup_symbols(frame, max_symbols=0) == []
    try:
        select_candidate_history_topup_symbols(frame, max_symbols=-1)
    except ValueError as exc:
        assert "max_symbols" in str(exc)
    else:
        raise AssertionError("negative top-up limit must be rejected")
