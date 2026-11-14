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
        "industry_codes": ["I01"],
    }, {
        "event_id": "e2", "published_at": "2026-01-03T08:00:00Z",
        "available_at": "2026-01-03T08:01:00Z", "source_url": "https://gov.cn/y",
        "source_id": "gov.cn", "title": "future", "catalyst_type": "industrial_policy",
        "industry_codes": ["I02"],
    }])
    visible, audit = visible_policy_events(events, decision_time="2026-01-02T12:00:00Z")
    assert visible.event_id.tolist() == ["e1"]
    assert audit["future_or_not_yet_available_rows"] == 1


def test_mainline_detects_ignition_peak_and_decay():
    dates = pd.date_range("2026-01-01", periods=8)
    rows = []
    # Five peer industries create a valid cross-sectional percentile each day.
    for i, day in enumerate(dates):
        for j in range(5):
            strength = [0, 1, 2, 4, 5, 5, 4, -4][i] if j == 0 else -j
            share = [0.02, 0.03, 0.04, 0.08, 0.08, 0.08, 0.08, 0.02][i] if j == 0 else 0.02
            rows.append({
                "trade_date": day, "industry_code": f"I{j}",
                "industry_return_5d": strength, "market_return_5d": 0,
                "turnover": share * 1000, "market_turnover": 1000,
            })
    out = classify_mainlines(pd.DataFrame(rows))
    focus = out[out.industry_code == "I0"].reset_index(drop=True)
    assert focus.mainline_phase.iloc[3] == "IGNITION"
    assert focus.mainline_phase.iloc[6] == "PEAK"
    assert focus.mainline_phase.iloc[7] == "DECAY"


def test_mainline_rejects_missing_values():
    frame = pd.DataFrame([{
        "trade_date": "2026-01-01", "industry_code": "I1",
        "industry_return_5d": None, "market_return_5d": 0,
        "turnover": 1, "market_turnover": 10,
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
