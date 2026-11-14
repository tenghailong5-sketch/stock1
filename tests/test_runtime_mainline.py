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
