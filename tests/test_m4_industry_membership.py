import pandas as pd
import pytest

from src.model.m4_industry_membership import (
    aggregate_pit_industry_turnover,
    select_pit_industry_membership,
)


DECISION = "2026-11-11T09:30:00+08:00"


def membership(rows):
    return pd.DataFrame(rows, columns=[
        "symbol", "industry_code", "valid_from", "valid_to",
        "available_time", "source_id", "source_revision", "snapshot_scope",
    ])


def row(symbol, industry, available=DECISION, start="2026-01-01",
        end=None, revision="v1"):
    return {
        "symbol": symbol,
        "industry_code": industry,
        "valid_from": start,
        "valid_to": end,
        "available_time": available,
        "source_id": "provider-x",
        "source_revision": revision,
        "snapshot_scope": "declared_universe",
    }


def select(frame, symbols=("000001", "000002"), trade_date="2026-11-10",
           decision=DECISION):
    return select_pit_industry_membership(
        frame, symbols, trade_date=trade_date, decision_time=decision,
        scope_id="declared_universe",
    )


def test_selects_latest_membership_revision_known_at_decision_time():
    frame = membership([
        row("000001", "OLD", available="2026-11-01T09:00:00+08:00", revision="r1"),
        row("000001", "NEW", available="2026-11-10T17:00:00+08:00", revision="r2"),
        row("000002", "BANK"),
    ])
    selected, audit = select(frame)
    assert selected.set_index("symbol").loc["000001", "industry_code"] == "NEW"
    assert audit["status"] == "ready"
    assert audit["historical_pit_eligible"] is True


def test_excludes_membership_revision_not_yet_available_at_decision():
    frame = membership([
        row("000001", "TECH", available="2026-11-12T09:00:00+08:00"),
        row("000002", "BANK"),
    ])
    selected, audit = select(frame)
    assert selected.empty
    assert audit["status"] == "incomplete_membership"
    assert audit["historical_pit_eligible"] is False


def test_validity_interval_is_start_inclusive_and_end_exclusive():
    frame = membership([
        row("000001", "TECH", start="2026-11-10", end="2026-11-11"),
        row("000002", "BANK", start="2026-11-10", end="2026-11-11"),
    ])
    selected, audit = select(frame, trade_date="2026-11-10")
    assert len(selected) == 2
    selected, audit = select(frame, trade_date="2026-11-11")
    assert selected.empty
    assert audit["status"] == "no_eligible_membership"


def test_incomplete_expected_universe_fails_closed():
    frame = membership([row("000001", "TECH")])
    selected, audit = select(frame)
    assert selected.empty
    assert audit["status"] == "incomplete_membership"
    assert audit["membership_coverage"] == 0.5


def test_conflicting_latest_membership_records_fail_closed():
    frame = membership([
        row("000001", "TECH"),
        row("000001", "BANK"),
        row("000002", "BANK"),
    ])
    selected, audit = select(frame)
    assert selected.empty
    assert audit["status"] == "ambiguous_latest_membership"


def test_rejects_naive_decision_time_and_invalid_validity_interval():
    frame = membership([row("000001", "TECH"), row("000002", "BANK")])
    with pytest.raises(ValueError, match="timezone-aware"):
        select(frame, decision="2026-11-11T09:30:00")
    invalid = membership([
        row("000001", "TECH", start="2026-11-10", end="2026-11-10"),
        row("000002", "BANK"),
    ])
    with pytest.raises(ValueError, match="valid_to"):
        select(invalid)


def test_rejects_future_trade_date_and_duplicate_expected_symbols():
    frame = membership([row("000001", "TECH"), row("000002", "BANK")])
    with pytest.raises(ValueError, match="after decision_time"):
        select(frame, trade_date="2026-11-12")
    with pytest.raises(ValueError, match="duplicates"):
        select_pit_industry_membership(
            frame, ["000001", "000001"], trade_date="2026-11-10",
            decision_time=DECISION, scope_id="declared_universe",
        )


def stock_bars(rows):
    return pd.DataFrame(rows, columns=[
        "symbol", "trade_date", "amount", "available_time",
        "historical_pit_eligible", "availability_basis",
    ])


def stock_bar(symbol, amount, *, pit=True, available=DECISION, basis="source_published_at"):
    return {
        "symbol": symbol,
        "trade_date": "2026-11-10",
        "amount": amount,
        "available_time": available,
        "historical_pit_eligible": pit,
        "availability_basis": basis,
    }


def aggregate(bars, members=None):
    if members is None:
        members = membership([row("000001", "TECH"), row("000002", "BANK")])
    return aggregate_pit_industry_turnover(
        bars, members, ["000001", "000002"],
        trade_date="2026-11-10", decision_time=DECISION,
        scope_id="declared_universe",
    )


def test_aggregates_turnover_only_after_complete_pit_bar_and_membership_join():
    bars = stock_bars([stock_bar("000001", 100), stock_bar("000002", 250)])
    result, audit = aggregate(bars)
    assert result.set_index("industry_code")["turnover"].to_dict() == {
        "BANK": 250, "TECH": 100,
    }
    assert audit["status"] == "ready"
    assert audit["historical_pit_eligible"] is True


def test_retrieval_time_historical_bars_are_not_pit_eligible():
    bars = stock_bars([
        stock_bar("000001", 100, pit=False, basis="retrieval_time_only"),
        stock_bar("000002", 250, pit=False, basis="retrieval_time_only"),
    ])
    result, audit = aggregate(bars)
    assert result.empty
    assert audit["status"] == "incomplete_pit_stock_bars"
    assert audit["stock_bar_coverage"] == 0
    assert audit["stock_bar_retrieval_only_symbols"] == 2
    assert audit["historical_pit_eligible"] is False


def test_incomplete_stock_bar_universe_blocks_industry_aggregation():
    bars = stock_bars([stock_bar("000001", 100)])
    result, audit = aggregate(bars)
    assert result.empty
    assert audit["status"] == "incomplete_pit_stock_bars"
    assert audit["stock_bar_coverage"] == 0.5


def test_incomplete_pit_membership_blocks_turnover_aggregation():
    bars = stock_bars([stock_bar("000001", 100), stock_bar("000002", 250)])
    members = membership([row("000001", "TECH")])
    result, audit = aggregate(bars, members)
    assert result.empty
    assert audit["status"] == "incomplete_pit_membership"
    assert audit["membership_coverage"] == 0.5


def test_future_stock_bar_revision_is_not_available_at_decision_time():
    bars = stock_bars([
        stock_bar("000001", 100, available="2026-11-12T09:00:00+08:00"),
        stock_bar("000002", 250),
    ])
    result, audit = aggregate(bars)
    assert result.empty
    assert audit["status"] == "incomplete_pit_stock_bars"
    assert audit["stock_bar_coverage"] == 0.5
