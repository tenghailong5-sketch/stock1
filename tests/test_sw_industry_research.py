import pandas as pd
import pytest

from src.model.sw_industry_research import select_sw_industry_research_snapshot


def history():
    return pd.DataFrame([
        {"symbol":"000001","industry_code":"480101","l1_code":"480000","l2_code":"480100",
         "valid_from":"2020-01-01","valid_to":"2024-01-01","available_time":"2026-11-15T00:00:00Z",
         "availability_basis":"retrieval_time_only","historical_pit_eligible":False},
        {"symbol":"000001","industry_code":"480301","l1_code":"480000","l2_code":"480300",
         "valid_from":"2024-01-01","valid_to":None,"available_time":"2026-11-15T00:00:00Z",
         "availability_basis":"retrieval_time_only","historical_pit_eligible":False},
        {"symbol":"600000","industry_code":"480201","l1_code":"480000","l2_code":"480200",
         "valid_from":"2020-01-01","valid_to":None,"available_time":"2026-11-15T00:00:00Z",
         "availability_basis":"retrieval_time_only","historical_pit_eligible":False},
    ])


def test_selects_effective_classification_on_requested_date():
    selected, audit = select_sw_industry_research_snapshot(
        history(), ["000001", "600000"], as_of_date="2025-06-01"
    )
    assert selected.set_index("symbol").loc["000001", "industry_code"] == "480301"
    assert audit["status"] == "ready_research_only"
    assert audit["coverage"] == 1.0


def test_historical_date_selects_previous_classification():
    selected, _ = select_sw_industry_research_snapshot(
        history(), ["000001"], as_of_date="2023-12-31"
    )
    assert selected.industry_code.tolist() == ["480101"]


def test_incomplete_coverage_is_reported_not_filled():
    selected, audit = select_sw_industry_research_snapshot(
        history(), ["000001", "000002"], as_of_date="2025-06-01"
    )
    assert selected.symbol.tolist() == ["000001"]
    assert audit["status"] == "incomplete_research_coverage"
    assert audit["coverage"] == 0.5


def test_duplicate_effective_rows_fail_closed():
    frame = pd.concat([history(), history().iloc[[1]]], ignore_index=True)
    selected, audit = select_sw_industry_research_snapshot(
        frame, ["000001"], as_of_date="2025-06-01"
    )
    assert selected.empty
    assert audit["status"] == "ambiguous_effective_membership"


def test_pit_eligible_row_is_rejected():
    frame = history()
    frame.loc[0, "historical_pit_eligible"] = True
    with pytest.raises(ValueError, match="PIT-eligible"):
        select_sw_industry_research_snapshot(
            frame, ["000001"], as_of_date="2023-01-01"
        )
