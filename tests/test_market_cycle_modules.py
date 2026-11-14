from pathlib import Path
import pandas as pd
import pytest

from src.model.market_cycle import classify_market_cycle
from src.model.policy_catalysts import visible_policy_events
from src.model.mainline_flows import classify_mainlines


def test_market_cycle_uses_observed_aggregates_and_marks_warmup():
    frame = pd.DataFrame({
        "trade_date": pd.date_range("2026-01-01", periods=5),
        "breadth": [0.2, 0.22, 0.25, 0.45, 0.72],
        "limit_up_count": [5, 6, 8, 20, 35],
        "limit_down_count": [30, 25, 20, 8, 3],
    })
    out = classify_market_cycle(frame)
    assert out.market_cycle.iloc[0] == "UNKNOWN"
    assert out.market_cycle.iloc[3] == "RECOVERY"
    assert out.market_cycle.iloc[-1] == "HEAT"
    assert not out.historical_pit_eligible.any()


def test_policy_events_exclude_not_yet_available_events():
    events = pd.DataFrame([{
        "event_id": "e1", "published_at": "2026-01-02T08:00:00Z",
        "available_at": "2026-01-02T08:01:00Z", "source_url": "https://gov.cn/x",
        "source_id": "gov.cn", "title": "policy", "catalyst_type": "industrial_policy",
        "industry_codes": ["I01"], "impact_direction": "positive",
        "confidence": 0.8, "confidence_basis": "formal_policy_text_and_explicit_industry_scope",
        "availability_basis": "publisher_timestamp",
    }, {
        "event_id": "e2", "published_at": "2026-01-03T08:00:00Z",
        "available_at": "2026-01-03T08:01:00Z", "source_url": "https://gov.cn/y",
        "source_id": "gov.cn", "title": "future", "catalyst_type": "industrial_policy",
        "industry_codes": ["I02"], "impact_direction": "positive",
        "confidence": 0.7, "confidence_basis": "formal_policy_text_and_explicit_industry_scope",
        "availability_basis": "publisher_timestamp",
    }])
    visible, audit = visible_policy_events(events, decision_time="2026-01-02T12:00:00Z")
    assert visible.event_id.tolist() == ["e1"]
    assert audit["future_or_not_yet_available_rows"] == 1


def test_mainline_detects_ignition_peak_and_decay():
    dates = pd.date_range("2026-01-01", periods=11)
    rows = []
    # Five peer industries create a valid cross-sectional percentile each day.
    for i, day in enumerate(dates):
        for j in range(5):
            strength = [0, 1, 2, 4, 5, 5, 5, 4, 3, -4, -5][i] if j == 0 else -j
            share = [0.02, 0.03, 0.04, 0.08, 0.09, 0.09, 0.08, 0.07, 0.06, 0.02, 0.01][i] if j == 0 else 0.02
            rows.append({
                "trade_date": day, "industry_code": f"I{j}",
                "industry_return_3d": strength, "market_return_3d": 0,
                "industry_return_5d": strength, "market_return_5d": 0,
                "turnover": share * 1000, "market_turnover": 1000,
                "expected_industry_count": 5,
            })
    out = classify_mainlines(pd.DataFrame(rows))
    focus = out[out.industry_code == "I0"].reset_index(drop=True)
    assert focus.mainline_phase.iloc[3] == "TRANSITION"
    assert focus.mainline_phase.iloc[4] == "IGNITION"
    assert focus.mainline_phase.iloc[8] == "PEAK"
    assert focus.mainline_phase.iloc[10] == "DECAY"
    assert focus.phase_transition.iloc[[4, 8, 10]].all()


def test_mainline_rejects_missing_values():
    frame = pd.DataFrame([{
        "trade_date": "2026-01-01", "industry_code": "I1",
        "industry_return_3d": None, "market_return_3d": 0,
        "industry_return_5d": None, "market_return_5d": 0,
        "turnover": 1, "market_turnover": 10, "expected_industry_count": 1,
    }])
    with pytest.raises(ValueError):
        classify_mainlines(frame)

def test_real_m3_archive_replays_market_emotion_history():
    from src.model.m3_assembler import assemble_m3_history

    archive = Path("data/forward_archive")
    result = assemble_m3_history(
        archive, index_symbol="000001",
        decision_time="2026-11-15T23:59:00+08:00",
    )
    assert result["status"] == "READY", result
    history = result["history"].rename(columns={
        "limit_up_cnt": "limit_up_count",
        "limit_down_cnt": "limit_down_count",
    })
    classified = classify_market_cycle(history)
    assert len(classified) >= 60
    assert classified.market_cycle.ne("UNKNOWN").sum() >= 55
    assert "ICE" in set(classified.market_cycle)
    assert "HEAT" in set(classified.market_cycle)
    assert classified.historical_pit_eligible.eq(False).all()

def test_mainline_rejects_incomplete_industry_cross_section():
    frame = pd.DataFrame([{
        "trade_date": "2026-01-01", "industry_code": "I1",
        "industry_return_3d": 1, "market_return_3d": 0,
        "industry_return_5d": 1, "market_return_5d": 0,
        "turnover": 100, "market_turnover": 1000,
        "expected_industry_count": 5,
    }])
    with pytest.raises(ValueError, match="peer universe"):
        classify_mainlines(frame)

def test_official_policy_archive_obeys_publisher_availability_dates():
    events = pd.read_json(
        "data/policy_catalysts/official_policy_events.jsonl", lines=True
    )
    before_mofcom_page = visible_policy_events(
        events, decision_time="2026-10-18T23:59:59+08:00"
    )[0]
    assert set(before_mofcom_page.event_id) == {
        "cn_policy_ai_manufacturing_20260109",
        "cn_policy_new_quality_productivity_20261014",
        "cn_policy_smart_factory_20260717",
        "cn_policy_new_battery_20260928",
        "cn_policy_smart_connected_ev_20260910",
        "cn_policy_ai_software_20260911",
        "cn_policy_electronic_info_plan_20260915",
        "cn_policy_pharma_industry_plan_20260918",
    }
    after_mofcom_page = visible_policy_events(
        events, decision_time="2026-10-20T23:59:59+08:00"
    )[0]
    assert "cn_policy_technology_commerce_20261019" in set(after_mofcom_page.event_id)
    assert "cn_policy_science_industry_integration_20261021" not in set(after_mofcom_page.event_id)
    after_all = visible_policy_events(
        events, decision_time="2026-10-22T23:59:59+08:00"
    )[0]
    assert len(after_all) == 10
    assert {
        "cn_policy_smart_factory_20260717",
        "cn_policy_new_battery_20260928",
        "cn_policy_smart_connected_ev_20260910",
        "cn_policy_ai_software_20260911",
        "cn_policy_electronic_info_plan_20260915",
        "cn_policy_pharma_industry_plan_20260918",
    }.issubset(set(after_all.event_id))
    assert after_all.confidence.between(0, 1).all()
    assert after_all.availability_basis.eq("publisher_timestamp").all()
    assert after_all.industry_codes.map(lambda values: len(values) > 0).all()

def test_raw_market_emotion_assembler_preserves_full_archive_history():
    from src.model.m3_assembler import assemble_market_emotion_history

    result = assemble_market_emotion_history(
        Path("data/forward_archive"),
        decision_time="2026-11-14T23:59:59+08:00",
        min_sessions=100,
    )
    assert result["status"] == "READY", result
    assert result["valid_sessions"] >= 100
    classified = classify_market_cycle(result["history"])
    assert classified.market_cycle.ne("UNKNOWN").sum() >= 95
    assert "ICE" in set(classified.market_cycle)
    assert "HEAT" in set(classified.market_cycle)
    assert "highest_observed_limit_up_streak" in result["history"].columns
    assert "strong_pool_count" in result["history"].columns
    assert result["history"].highest_observed_limit_up_streak.notna().sum() >= 1
    assert classified.historical_pit_eligible.eq(False).all()


def test_emotion_assembler_merges_supplemental_fields_without_replacing_primary_counts(tmp_path):
    import json
    from src.model.m3_assembler import assemble_market_emotion_history

    breadth_dir = tmp_path / "market_breadth"
    sentiment_dir = tmp_path / "market_sentiment"
    breadth_dir.mkdir()
    sentiment_dir.mkdir()
    (breadth_dir / "2026-01-05.jsonl").write_text(json.dumps({
        "snapshot_time": "2026-01-05T08:00:00Z",
        "available_time": "2026-01-05T08:00:00Z",
        "breadth": 0.55, "breadth_coverage": 1.0,
        "total_amount": 1000, "total_amount_status": "complete_provider_snapshot_sum",
        "source_id": "primary.breadth",
    }) + "\n", encoding="utf-8")
    rows = [
        {
            "trade_date": "2026-01-05",
            "capture_time": "2026-01-05T08:00:00Z",
            "available_time": "2026-01-05T08:00:00Z",
            "limit_up_count": 10, "limit_down_count": 2,
            "source_id": "kaipanhong.market_emotion_kph",
        },
        {
            "trade_date": "2026-01-05",
            "capture_time": "2026-01-05T09:00:00Z",
            "available_time": "2026-01-05T09:00:00Z",
            "limit_up_count": 12, "limit_down_count": 1,
            "highest_observed_limit_up_streak": 4,
            "strong_pool_count": 50,
            "source_id": "secondary.limit_pool",
        },
    ]
    (sentiment_dir / "2026-01-05.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    result = assemble_market_emotion_history(
        tmp_path, decision_time="2026-01-05T18:00:00+08:00", min_sessions=1,
    )
    assert result["status"] == "READY", result
    row = result["history"].iloc[0]
    assert row.limit_up_count == 10
    assert row.limit_down_count == 2
    assert row.highest_observed_limit_up_streak == 4
    assert row.strong_pool_count == 50
    assert row.sentiment_source_id == "kaipanhong.market_emotion_kph"
    assert row.sentiment_context_source_id == "secondary.limit_pool"
    assert pd.Timestamp(row.available_time) == pd.Timestamp("2026-01-05T09:00:00Z")
