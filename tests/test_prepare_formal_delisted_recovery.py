from __future__ import annotations

import pytest

from scripts.prepare_formal_delisted_recovery import sanitize_catalog


def test_sanitize_catalog_keeps_only_formal_stocks_and_latest_terminal():
    existing = {
        "delisted": {
            "600001.SH": "2020-01-02",
            "600002.SH": "2021-01-01",
            "600004.SH": "2022-01-01",
        },
        "never_issued": ["000001.SZ"],
    }
    raw = {
        "delisted": {
            "600001.SH": "2020-01-03",
            "600003.SH": "2023-01-01",
            "600005.SH": "2024-01-01",
        },
        "never_issued": ["000002.SZ"],
        "schema_version": 1,
    }
    formal = {
        "600001.SH": "2020-02-01",
        "600002.SH": "2021-02-01",
        "600003.SH": "2023-02-01",
    }
    listing = {"600001.SH", "600002.SH", "600003.SH", "600004.SH"}

    sanitized, audit = sanitize_catalog(existing, raw, formal, listing)

    assert sanitized["delisted"] == {
        "600001.SH": "2020-01-03",
        "600002.SH": "2021-01-01",
        "600003.SH": "2023-01-01",
    }
    assert sanitized["never_issued"] == ["000002.SZ"]
    assert audit["raw_sina_probe_hits"] == 3
    assert audit["formal_delisted_hits_in_raw_catalog"] == 2
    assert audit["currently_listed_hits_quarantined"] == 0
    assert audit["unmatched_identity_hits_quarantined"] == 1
    assert audit["formal_delistings_not_yet_in_raw_catalog"] == 1
    assert audit["historical_pit_eligible"] is False


def test_sanitize_catalog_quarantines_currently_listed_probe_hits():
    raw = {
        "delisted": {
            "600001.SH": "2020-01-03",
            "600002.SH": "2024-01-01",
        },
        "never_issued": [],
    }
    formal = {"600001.SH": "2020-02-01"}
    listing = {"600001.SH", "600002.SH"}

    sanitized, audit = sanitize_catalog({}, raw, formal, listing)

    assert sanitized["delisted"] == {"600001.SH": "2020-01-03"}
    assert audit["currently_listed_hits_quarantined"] == 1
    assert audit["unmatched_identity_hits_quarantined"] == 0


def test_sanitize_catalog_preserves_formal_existing_terminals_when_not_in_raw():
    existing = {"delisted": {"600001.SH": "2020-01-03"}, "never_issued": []}
    raw = {"delisted": {}, "never_issued": []}

    sanitized, audit = sanitize_catalog(
        existing,
        raw,
        {"600001.SH": "2020-02-01"},
        {"600001.SH"},
    )

    assert sanitized["delisted"] == {"600001.SH": "2020-01-03"}
    assert audit["formal_delistings_not_yet_in_raw_catalog"] == 1


def test_sanitize_catalog_requires_nonempty_formal_identity_set():
    with pytest.raises(ValueError, match="empty"):
        sanitize_catalog({}, {"delisted": {}, "never_issued": []}, {}, set())


def test_sanitize_catalog_rejects_invalid_terminal_date():
    with pytest.raises(ValueError, match="invalid terminal date"):
        sanitize_catalog(
            {},
            {"delisted": {"600001.SH": "not-a-date"}, "never_issued": []},
            {"600001.SH": "2020-02-01"},
            {"600001.SH"},
        )
