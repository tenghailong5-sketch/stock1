import json

import pandas as pd
import pytest

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
            "close": [100.0 + i for i in range(len(dates))],
            "return_3": [0.02] * len(dates), "return_5": [0.03] * len(dates),
            "amount": [1_000_000.0] * len(dates),
        })

    monkeypatch.setattr(
        runtime_mainline, "fetch_sw_industry_index_history_native", fetch_history
    )
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
        runtime_mainline, "fetch_sw_industry_index_history_native", unavailable
    )
    monkeypatch.setattr(
        runtime_mainline, "fetch_sw_api_industry_index_history", unavailable
    )
    try:
        runtime_mainline._assemble_industry_panel(
            decision_time=pd.Timestamp("2026-02-01T08:00:00Z"),
            archive_root=tmp_path,
        )
    except RuntimeError as exc:
        assert "unavailable from alternate/live/cache providers" in str(exc)
    else:
        raise AssertionError("an unavailable first batch should abort early")
    assert len(calls) == 8


def test_cached_market_emotion_uses_completed_exchange_sessions_only(monkeypatch, tmp_path):
    dates = pd.bdate_range("2026-01-05", periods=4)
    # A Saturday provider snapshot must not become the latest market session.
    history = pd.DataFrame({
        "trade_date": list(dates) + [pd.Timestamp("2026-01-10")],
        "breadth": [0.20, 0.25, 0.35, 0.62, 0.01],
        "limit_up_count": [8, 9, 12, 22, 0],
        "limit_down_count": [25, 20, 18, 6, 80],
    })
    index_dir = tmp_path / "index_daily"
    index_dir.mkdir()
    (index_dir / "000001.jsonl").write_text(
        "\n".join(json.dumps({
            "trade_date": day.strftime("%Y-%m-%d"),
            "close": 3000 + i,
            "available_time": "2026-01-20T07:00:00Z",
        }) for i, day in enumerate(dates)) + "\n",
        encoding="utf-8",
    )
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
    assert audit["latest_trade_date"] == "2026-01-08"
    assert audit["market_calendar_latest_trade_date"] == "2026-01-08"
    assert audit["valid_sessions"] == 4
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


def test_forward_industry_flow_archive_warms_and_computes_three_session_change(tmp_path):
    from src.model.forward_industry_flow import attach_forward_industry_flow_changes

    archive = tmp_path / "forward"
    base = pd.Timestamp("2026-11-09T16:00:00+08:00")
    latest = None
    for offset, share in enumerate([0.10, 0.12, 0.15, 0.18]):
        day = base + pd.Timedelta(days=offset)
        while day.weekday() >= 5:
            day += pd.Timedelta(days=1)
        panel = pd.DataFrame({
            "industry_name": ["计算机", "传媒"],
            "trade_date": [day.normalize(), day.normalize()],
            "turnover_share": [share, 0.20],
        })
        latest, audit = attach_forward_industry_flow_changes(
            panel, decision_time=day, archive_root=archive,
        )
    row = latest.set_index("industry_name").loc["计算机"]
    assert row["turnover_share_change_3d"] == pytest.approx(0.08)
    assert row["turnover_share_change_1d"] == pytest.approx(0.03)
    assert audit["status"] == "READY_FORWARD_FLOW_RESEARCH"
    assert not audit["historical_pit_eligible"]
    assert not audit["trading_authorization"]


def test_forward_industry_flow_archive_does_not_persist_weekend_snapshots(tmp_path):
    from src.model.forward_industry_flow import attach_forward_industry_flow_changes

    panel = pd.DataFrame({
        "industry_name": ["计算机"],
        "trade_date": [pd.Timestamp("2026-11-15")],
        "turnover_share": [0.2],
    })
    result, audit = attach_forward_industry_flow_changes(
        panel,
        decision_time=pd.Timestamp("2026-11-15T16:00:00+08:00"),
        archive_root=tmp_path,
    )
    assert not audit["archived"]
    assert audit["status"] == "WAITING_COMPLETED_SESSION_ARCHIVE"
    assert not (tmp_path / "industry_flow_daily.jsonl").exists()


def test_full_emotion_assembly_excludes_weekend_provider_snapshot(monkeypatch, tmp_path):
    sessions = pd.bdate_range("2026-01-05", periods=4)
    weekend = pd.Timestamp("2026-01-10")
    history = pd.DataFrame({
        "trade_date": list(sessions) + [weekend],
        "breadth": [0.2, 0.25, 0.3, 0.4, 0.01],
        "limit_up_count": [8, 9, 12, 20, 0],
        "limit_down_count": [25, 20, 18, 6, 80],
        "available_time": [pd.Timestamp("2026-01-20T07:00:00Z")] * 5,
    })
    calendar = pd.DataFrame({"trade_date": sessions})
    monkeypatch.setattr(
        runtime_mainline, "assemble_market_emotion_history",
        lambda *args, **kwargs: {"status": "READY", "history": history.copy()},
    )
    monkeypatch.setattr(runtime_mainline, "_read_index_archive", lambda *args, **kwargs: calendar.copy())
    monkeypatch.setattr(
        runtime_mainline, "fetch_market_emotion_pool_history",
        lambda dates, **kwargs: pd.DataFrame({
            "trade_date": dates,
            "available_time": [pd.Timestamp("2026-01-20T07:30:00Z")] * len(dates),
            "pool_history_status": ["READY"] * len(dates),
        }),
    )
    monkeypatch.setattr(
        runtime_mainline, "classify_market_cycle",
        lambda frame: frame.assign(market_cycle="TRANSITION"),
    )
    monkeypatch.setattr(
        runtime_mainline, "audit_emotion_history",
        lambda frame: {"sessions": len(frame)},
    )
    cycles, audit = runtime_mainline._assemble_emotion(
        decision_time=pd.Timestamp("2026-01-20T08:00:00Z"),
        archive_root=tmp_path,
    )
    assert cycles.trade_date.max() == sessions[-1]
    assert audit["latest_trade_date"] == "2026-01-08"
    assert len(cycles) == len(sessions)


def test_industry_panel_uses_recent_cached_history_when_both_live_routes_fail(monkeypatch, tmp_path):
    dates = pd.bdate_range(end="2026-02-13", periods=30)
    codes = [f"801{i:03d}" for i in range(20)]
    universe_dir = tmp_path / "sw_industry_universe"
    history_dir = tmp_path / "industry_index_history"
    index_dir = tmp_path / "index_daily"
    universe_dir.mkdir()
    history_dir.mkdir()
    index_dir.mkdir()
    (universe_dir / "2026-02-14.jsonl").write_text(
        "\\n".join(json.dumps({
            "industry_code": code,
            "industry_name": f"industry-{i}",
            "observed_at": "2026-02-14T07:00:00Z",
        }) for i, code in enumerate(codes)) + "\\n",
        encoding="utf-8",
    )
    for i, code in enumerate(codes):
        pd.DataFrame({
            "trade_date": dates.strftime("%Y-%m-%d"),
            "industry_code": code,
            "close": [100.0 + j + i * 0.1 for j in range(len(dates))],
            "amount": [1_000_000.0] * len(dates),
            "available_time": ["2026-02-14T07:00:00Z"] * len(dates),
        }).to_json(history_dir / f"{code}.jsonl", orient="records", lines=True)
    (index_dir / "000001.jsonl").write_text(
        "\\n".join(json.dumps({
            "trade_date": day.strftime("%Y-%m-%d"),
            "close": 3000.0 + i * 2,
            "available_time": "2026-02-14T07:00:00Z",
        }) for i, day in enumerate(dates)) + "\\n",
        encoding="utf-8",
    )

    def unavailable(*args, **kwargs):
        raise RuntimeError("live SW providers unavailable")

    monkeypatch.setattr(runtime_mainline, "fetch_sw_industry_index_history_native", unavailable)
    monkeypatch.setattr(runtime_mainline, "fetch_sw_api_industry_index_history", unavailable)
    panel, audit = runtime_mainline._assemble_industry_panel(
        decision_time=pd.Timestamp("2026-02-15T08:00:00Z"),
        archive_root=tmp_path,
    )
    assert audit["universe_source"] == "archived_SW_universe"
    assert audit["industry_count"] == 20
    assert audit["available_industries"] == 20
    assert audit["minimum_industries_per_date"] == 18
    assert audit["latest_trade_date"] == "2026-02-13"
    assert panel.industry_code.nunique() == 20
    assert not audit["historical_pit_eligible"]
