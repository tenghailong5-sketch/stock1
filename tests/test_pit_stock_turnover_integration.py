import pandas as pd

from src.model.mainline_flows import attach_pit_stock_turnover_concentration


def base():
    return pd.DataFrame([
        {"trade_date": "2026-11-10", "industry_code": "BANK", "expected_industry_count": 2, "turnover": 800},
        {"trade_date": "2026-11-10", "industry_code": "TECH", "expected_industry_count": 2, "turnover": 200},
    ])


def pit_rows():
    return pd.DataFrame([
        {
            "trade_date": "2026-11-10", "industry_code": "BANK", "turnover": 300,
            "available_time": "2026-11-10T17:00:00+08:00",
            "decision_time": "2026-11-11T09:30:00+08:00",
            "availability_basis": "source_published_at", "historical_pit_eligible": True,
        },
        {
            "trade_date": "2026-11-10", "industry_code": "TECH", "turnover": 700,
            "available_time": "2026-11-10T17:00:00+08:00",
            "decision_time": "2026-11-11T09:30:00+08:00",
            "availability_basis": "source_published_at", "historical_pit_eligible": True,
        },
    ])


def test_attaches_stock_turnover_share_without_replacing_index_turnover():
    result = attach_pit_stock_turnover_concentration(base(), pit_rows())
    rows = result.set_index("industry_code")
    assert rows.loc["BANK", "turnover"] == 800
    assert rows.loc["TECH", "turnover"] == 200
    assert rows.loc["BANK", "stock_turnover"] == 300
    assert rows.loc["TECH", "stock_turnover"] == 700
    assert rows.loc["BANK", "stock_turnover_share"] == 0.3
    assert rows.loc["TECH", "stock_turnover_share"] == 0.7
    assert rows["stock_turnover_pit_status"].eq("ready").all()


def test_incomplete_industry_coverage_fails_closed():
    result = attach_pit_stock_turnover_concentration(base(), pit_rows().iloc[:1])
    assert result["stock_turnover_share"].isna().all()
    assert result["stock_turnover_pit_status"].eq("incomplete_industry_coverage").all()


def test_retrieval_time_only_data_fails_closed():
    pit = pit_rows()
    pit["historical_pit_eligible"] = False
    pit["availability_basis"] = "retrieval_time_only"
    result = attach_pit_stock_turnover_concentration(base(), pit)
    assert result["stock_turnover_share"].isna().all()
    assert result["stock_turnover_pit_status"].eq("pit_lineage_rejected").all()


def test_future_available_revision_fails_closed():
    pit = pit_rows()
    pit.loc[0, "available_time"] = "2026-11-12T09:00:00+08:00"
    result = attach_pit_stock_turnover_concentration(base(), pit)
    assert result["stock_turnover_share"].isna().all()
    assert result["stock_turnover_pit_status"].eq("pit_lineage_rejected").all()
