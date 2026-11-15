import pandas as pd
import pytest

from src.providers.cnequity_mainline import (
    build_cnequity_mainline_inputs,
    load_cnequity_mainline_inputs,
)


def _frames():
    dates = pd.date_range("2026-01-01", periods=6, freq="B")
    industry_rows = []
    for industry_number in range(10):
        code = f"8010{industry_number:02d}"
        multiplier = (industry_number + 1) / 10.0
        for index, day in enumerate(dates):
            industry_rows.append({
                "trade_date": day,
                "industry_code": code,
                "level": "L1",
                "weighting": "amount",
                "ret": 0.01 * multiplier,
                "amount": 100.0 * multiplier + index,
                "n_members": 100,
                "n_priced": 95,
                "n_excluded": 5,
                "source": "derived",
                "data_version": "v1",
                "fetched_at": "2026-01-10T08:00:00Z",
            })
    market_rows = [{
        "symbol": "000001.SH",
        "trade_date": day,
        "close": 100.0 + index,
        "frequency": "1d",
        "source": "exchange",
        "data_version": "v1",
        "fetched_at": "2026-01-10T08:00:00Z",
    } for index, day in enumerate(dates)]
    return pd.DataFrame(industry_rows), pd.DataFrame(market_rows)


def test_builds_amount_weighted_level1_mainline_panel_as_research_only():
    industry, market = _frames()
    panel, audit = build_cnequity_mainline_inputs(
        industry, market, expected_industry_count=10,
    )
    assert audit["status"] == "READY_RESEARCH_ONLY"
    assert audit["industry_count"] == 10
    assert audit["trade_date_count"] == 1
    assert len(panel) == 10
    assert panel["historical_pit_eligible"].eq(False).all()
    assert panel["research_only"].all()
    assert panel["trading_authorization"].eq(False).all()
    assert panel["availability_basis"].eq(
        "historical_reconstruction_from_cnequity_lake"
    ).all()
    assert panel["market_turnover"].gt(0).all()
    assert panel["member_coverage"].eq(0.95).all()
    assert panel[["n_members", "n_priced", "n_excluded"]].notna().all().all()
    assert audit["market_source"] == ["exchange"]


def test_rejects_duplicate_industry_date_and_naive_provenance():
    industry, market = _frames()
    industry = pd.concat([industry, industry.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        build_cnequity_mainline_inputs(industry, market, expected_industry_count=10)

    industry, market = _frames()
    industry["fetched_at"] = "2026-01-10T08:00:00"
    with pytest.raises(ValueError, match="timezone-aware"):
        build_cnequity_mainline_inputs(industry, market, expected_industry_count=10)


def test_loader_reads_industry_and_explicit_benchmark_with_data_root():
    industry, market = _frames()
    calls = []

    def fake_loader(dataset, **kwargs):
        calls.append((dataset, kwargs))
        return industry if dataset == "industry_index" else market

    panel, audit = load_cnequity_mainline_inputs(
        start="2026-01-01", end="2026-01-31",
        data_root="/tmp/cnequity", loader=fake_loader,
        expected_industry_count=10,
    )
    assert [name for name, _ in calls] == ["industry_index", "index_bars"]
    assert calls[0][1]["data_root"] == "/tmp/cnequity"
    assert calls[1][1]["symbols"] == ["000001.SH"]
    assert audit["provider"] == "CNEquity"
    assert len(panel) == 20

def test_excludes_low_priced_member_coverage_without_hiding_the_count():
    industry, market = _frames()
    industry.loc[industry["industry_code"].eq("801001"), "n_priced"] = 70
    industry.loc[industry["industry_code"].eq("801001"), "n_excluded"] = 30
    panel, audit = build_cnequity_mainline_inputs(
        industry, market, expected_industry_count=10,
        min_priced_coverage=0.80,
    )
    assert "801001" not in set(panel["industry_code"])
    assert panel["industry_code"].nunique() == 9
    assert audit["low_coverage_rows_excluded"] == 6


def test_rejects_inconsistent_industry_member_counts():
    industry, market = _frames()
    industry.loc[0, "n_excluded"] = 4
    with pytest.raises(ValueError, match="inconsistent"):
        build_cnequity_mainline_inputs(industry, market, expected_industry_count=10)

def test_drops_an_industry_with_gapped_sessions_instead_of_bridging_returns():
    industry, market = _frames()
    industry = industry.loc[
        ~(
            industry["industry_code"].eq("801001")
            & industry["trade_date"].eq(industry["trade_date"].iloc[2])
        )
    ].copy()
    panel, audit = build_cnequity_mainline_inputs(
        industry, market, expected_industry_count=10,
    )
    assert "801001" not in set(panel["industry_code"])
    assert audit["incomplete_industries_excluded"] == 1
    assert panel["industry_code"].nunique() == 9

