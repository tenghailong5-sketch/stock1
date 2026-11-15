import json

import pandas as pd

from src.model import runtime_mainline


def test_runtime_mainline_scan_connects_phases_and_evidence(monkeypatch, tmp_path):
    dates = pd.bdate_range("2026-01-05", periods=8)
    codes = [f"{i:06d}" for i in range(1, 21)]
    rows = []
    for day_index, day in enumerate(dates):
        amounts = [(i + 1) * 1_000_000 for i in range(len(codes))]
        total = sum(amounts)
        for i, code in enumerate(codes):
            rows.append({
                "trade_date": day,
                "industry_code": code,
                "industry_name": f"industry-{i}",
                "industry_return_3d": 0.02 + i * 0.0001 + day_index * 0.0002,
                "market_return_3d": 0.01,
                "industry_return_5d": 0.03 + i * 0.0001 + day_index * 0.0003,
                "market_return_5d": 0.015,
                "turnover": amounts[i],
                "market_turnover": total,
                "expected_industry_count": len(codes),
            })
    panel = pd.DataFrame(rows)
    emotion = pd.DataFrame({
        "trade_date": dates,
        "market_cycle": ["ICE", "RECOVERY", "TRANSITION", "TRANSITION",
                         "HEAT", "EXHAUSTION", "TRANSITION", "TRANSITION"],
    })
    monkeypatch.setattr(
        runtime_mainline, "_assemble_industry_panel",
        lambda **kwargs: (panel.copy(), {
            "industry_count": 20, "complete_trade_dates": 8,
            "latest_trade_date": dates[-1].strftime("%Y-%m-%d"),
            "historical_pit_eligible": False,
        }),
    )
    monkeypatch.setattr(
        runtime_mainline, "_assemble_emotion",
        lambda **kwargs: (emotion.copy(), {"sessions": 8, "fields": {}, "historical_pit_eligible": False}),
    )
    policy_path = tmp_path / "policy.jsonl"
    policy_path.write_text(json.dumps({
        "event_id": "test-event",
        "published_at": "2026-01-05T08:00:00Z",
        "available_at": "2026-01-05T08:00:00Z",
        "source_url": "https://example.com/policy",
        "source_id": "example.com",
        "title": "test policy",
        "catalyst_type": "test",
        "industry_codes": [codes[0]],
        "impact_direction": "positive",
        "confidence": 0.9,
        "confidence_basis": "unit test",
        "availability_basis": "publisher_timestamp",
    }) + "\n", encoding="utf-8")

    latest, audit = runtime_mainline.scan_runtime_mainlines(
        decision_time=pd.Timestamp("2026-01-20T08:00:00Z"),
        archive_root=tmp_path,
        policy_path=policy_path,
    )

    assert audit["status"] == "READY_DIAGNOSTIC_ONLY"
    assert audit["latest_industry_rows"] == 20
    assert len(latest) == 20
    assert "mainline_phase_gated" in latest
    assert "startup_evidence_status" in latest
    assert latest["turnover_share_basis"].eq(
        "relative_share_of_sw_first_level_industry_index_turnover"
    ).all()
    assert not audit["historical_pit_eligible"]
    assert not audit["policy"]["archive_complete"]


def test_runtime_mainline_scan_fails_closed_when_sources_unavailable(monkeypatch):
    def unavailable(**kwargs):
        raise RuntimeError("provider unavailable")

    monkeypatch.setattr(runtime_mainline, "_assemble_industry_panel", unavailable)
    latest, audit = runtime_mainline.scan_runtime_mainlines(
        decision_time=pd.Timestamp("2026-01-20T08:00:00Z")
    )
    assert latest.empty
    assert audit["status"] == "UNAVAILABLE"
    assert "provider unavailable" in audit["reason"]
    assert not audit["formal_stock_score_eligible"]



def test_runtime_mainline_keeps_industry_and_emotion_when_policy_archive_is_missing(monkeypatch, tmp_path):
    dates = pd.bdate_range("2026-01-05", periods=8)
    codes = [f"{i:06d}" for i in range(1, 21)]
    rows = []
    for day_index, day in enumerate(dates):
        amounts = [(i + 1) * 1_000_000 for i in range(len(codes))]
        total = sum(amounts)
        for i, code in enumerate(codes):
            rows.append({
                "trade_date": day, "industry_code": code,
                "industry_name": f"industry-{i}",
                "industry_return_3d": 0.02 + i * 0.0001 + day_index * 0.0002,
                "market_return_3d": 0.01,
                "industry_return_5d": 0.03 + i * 0.0001 + day_index * 0.0003,
                "market_return_5d": 0.015,
                "turnover": amounts[i], "market_turnover": total,
                "expected_industry_count": len(codes),
            })
    panel = pd.DataFrame(rows)
    emotion = pd.DataFrame({
        "trade_date": dates,
        "market_cycle": ["ICE", "RECOVERY", "TRANSITION", "TRANSITION",
                         "HEAT", "EXHAUSTION", "TRANSITION", "TRANSITION"],
    })
    monkeypatch.setattr(
        runtime_mainline, "_assemble_industry_panel",
        lambda **kwargs: (panel.copy(), {"industry_count": 20, "complete_trade_dates": 8,
                                         "latest_trade_date": dates[-1].strftime("%Y-%m-%d"),
                                         "historical_pit_eligible": False}),
    )
    monkeypatch.setattr(
        runtime_mainline, "_assemble_emotion",
        lambda **kwargs: (emotion.copy(), {"sessions": 8, "fields": {},
                                           "historical_pit_eligible": False}),
    )

    latest, audit = runtime_mainline.scan_runtime_mainlines(
        decision_time=pd.Timestamp("2026-01-20T08:00:00Z"),
        archive_root=tmp_path,
        policy_path=tmp_path / "missing-policy.jsonl",
    )

    assert audit["status"] == "READY_DIAGNOSTIC_ONLY"
    assert audit["policy"]["status"] == "ARCHIVE_MISSING_UNKNOWN"
    assert audit["policy"]["visible_events"] == 0
    assert len(latest) == 20
    assert not audit["historical_pit_eligible"]



def test_industry_panel_keeps_ninety_percent_when_two_providers_fail(monkeypatch, tmp_path):
    dates = pd.bdate_range("2026-01-05", periods=8)
    codes = [f"I{i:02d}" for i in range(20)]
    groups = pd.DataFrame({
        "industry_code": codes, "industry_name": [f"industry-{i}" for i in range(20)]
    })
    monkeypatch.setattr(
        runtime_mainline, "fetch_sw_api_first_level_index_list",
        lambda **kwargs: groups.copy(),
    )

    def fetch_history(code, **kwargs):
        if code in {"I18", "I19"}:
            raise RuntimeError("temporary provider timeout")
        return pd.DataFrame({
            "trade_date": dates, "industry_code": code,
            "return_3": [0.02] * len(dates), "return_5": [0.03] * len(dates),
            "amount": [1_000_000.0] * len(dates),
        })

    monkeypatch.setattr(
        runtime_mainline, "fetch_sw_api_industry_index_history", fetch_history
    )
    index_dir = tmp_path / "index_daily"
    index_dir.mkdir()
    (index_dir / "000001.jsonl").write_text(
        "\n".join(json.dumps({
            "trade_date": day.strftime("%Y-%m-%d"),
            "close": 3000 + i * 2,
            "available_time": "2026-02-01T07:00:00Z",
        }) for i, day in enumerate(dates)) + "\n",
        encoding="utf-8",
    )

    panel, audit = runtime_mainline._assemble_industry_panel(
        decision_time=pd.Timestamp("2026-02-01T08:00:00Z"),
        archive_root=tmp_path,
    )

    assert audit["industry_count"] == 20
    assert audit["minimum_industries_per_date"] == 18
    assert audit["failed_industry_count"] == 2
    assert audit["available_industries"] == 18
    assert panel.groupby("trade_date").industry_code.nunique().eq(18).all()



def test_industry_panel_aborts_remaining_batches_after_initial_transport_outage(monkeypatch, tmp_path):
    import requests

    groups = pd.DataFrame({
        "industry_code": [f"I{i:02d}" for i in range(20)],
        "industry_name": [f"industry-{i}" for i in range(20)],
    })
    calls = []
    monkeypatch.setattr(
        runtime_mainline, "fetch_sw_api_first_level_index_list",
        lambda **kwargs: groups.copy(),
    )

    def unavailable(code, **kwargs):
        calls.append(code)
        raise requests.exceptions.ReadTimeout("SW endpoint unavailable")

    monkeypatch.setattr(
        runtime_mainline, "fetch_sw_api_industry_index_history", unavailable
    )
    try:
        runtime_mainline._assemble_industry_panel(
            decision_time=pd.Timestamp("2026-02-01T08:00:00Z"),
            archive_root=tmp_path,
        )
    except RuntimeError as exc:
        assert "aborted remaining industry requests early" in str(exc)
    else:
        raise AssertionError("an unavailable first batch should abort early")
    assert len(calls) == 4


def test_cached_market_emotion_uses_archive_only_and_returns_latest_cycle(monkeypatch, tmp_path):
    dates = pd.bdate_range("2026-01-05", periods=4)
    history = pd.DataFrame({
        "trade_date": dates,
        "breadth": [0.20, 0.25, 0.35, 0.62],
        "limit_up_count": [8, 9, 12, 22],
        "limit_down_count": [25, 20, 18, 6],
    })
    calls = []
    def cached_history(*args, **kwargs):
        calls.append((args, kwargs))
        return {"status": "READY", "valid_sessions": 4, "history": history.copy()}
    monkeypatch.setattr(
        runtime_mainline, "assemble_market_emotion_history", cached_history
    )
    latest, audit = runtime_mainline.scan_cached_market_emotion(
        decision_time=pd.Timestamp("2026-01-20T08:00:00Z"),
        archive_root=tmp_path, min_sessions=4,
    )
    assert len(calls) == 1
    assert audit["status"] == "READY_CACHED_RESEARCH_ONLY"
    assert audit["market_cycle"] == "RECOVERY"
    assert latest.iloc[0]["market_cycle"] == "RECOVERY"
    assert audit["cache_only"] is True
    assert audit["historical_pit_eligible"] is False
    assert audit["trading_authorization"] is False


def test_cached_market_emotion_missing_archive_degrades_to_unknown(monkeypatch, tmp_path):
    monkeypatch.setattr(
        runtime_mainline, "assemble_market_emotion_history",
        lambda *args, **kwargs: {
            "status": "INSUFFICIENT_DATA", "valid_sessions": 2,
            "history": pd.DataFrame(),
        },
    )
    latest, audit = runtime_mainline.scan_cached_market_emotion(
        decision_time=pd.Timestamp("2026-01-20T08:00:00Z"),
        archive_root=tmp_path, min_sessions=4,
    )
    assert latest.empty
    assert audit["status"] == "UNKNOWN_NO_CACHED_EMOTION"
    assert audit["market_cycle"] == "UNKNOWN"
    assert not audit["trading_authorization"]
