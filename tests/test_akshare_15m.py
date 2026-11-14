import pandas as pd
import pytest

from src.providers.akshare_15m import normalize_15m_bars, confirm_15m_breakout


def sample():
    # Four completed reference bars followed by a breakout bar.
    return pd.DataFrame([
        {"时间": "2026-11-13 10:00:00", "开盘": 10.0, "最高": 10.2, "最低": 9.9, "收盘": 10.1, "成交量": 100, "成交额": 1000},
        {"时间": "2026-11-13 10:15:00", "开盘": 10.1, "最高": 10.3, "最低": 10.0, "收盘": 10.2, "成交量": 110, "成交额": 1100},
        {"时间": "2026-11-13 10:30:00", "开盘": 10.2, "最高": 10.4, "最低": 10.1, "收盘": 10.3, "成交量": 90, "成交额": 900},
        {"时间": "2026-11-13 10:45:00", "开盘": 10.3, "最高": 10.35, "最低": 10.2, "收盘": 10.25, "成交量": 100, "成交额": 1000},
        {"时间": "2026-11-13 11:00:00", "开盘": 10.4, "最高": 10.6, "最低": 10.35, "收盘": 10.55, "成交量": 150, "成交额": 1500},
    ])


def test_confirms_only_closed_breakout_bar_with_volume_expansion():
    bars = normalize_15m_bars(
        sample(), symbol="600000", captured_at="2026-11-13T03:00:00Z"
    )
    result = confirm_15m_breakout(bars, lookback_bars=4, minimum_volume_multiple=1.0)
    assert result["entry_ready"] is True
    assert result["status"] == "CONFIRMED_15M_BREAKOUT"
    assert result["historical_pit_eligible"] is False
    assert result["checks"]["close_above_prior_high"] is True


def test_partial_current_bar_is_excluded():
    bars = normalize_15m_bars(
        sample(), symbol="600000", captured_at="2026-11-13T02:10:00Z"
    )
    assert len(bars) == 4
    assert confirm_15m_breakout(bars)["entry_ready"] is False


def test_no_breakout_does_not_confirm():
    raw = sample()
    raw.loc[4, "收盘"] = 10.3
    raw.loc[4, "开盘"] = 10.3
    bars = normalize_15m_bars(
        raw, symbol="600000", captured_at="2026-11-13T03:00:00Z"
    )
    result = confirm_15m_breakout(bars)
    assert result["entry_ready"] is False
    assert "close_above_prior_high" in result["reason"]


def test_naive_capture_time_is_rejected():
    with pytest.raises(ValueError, match="timezone-aware"):
        normalize_15m_bars(sample(), symbol="600000", captured_at="2026-11-13 11:20:00")


def test_duplicate_bar_timestamps_are_rejected():
    raw = sample()
    raw.loc[4, "时间"] = raw.loc[3, "时间"]
    with pytest.raises(ValueError, match="duplicate"):
        normalize_15m_bars(raw, symbol="600000", captured_at="2026-11-13T03:00:00Z")


def test_invalid_ohlc_is_rejected():
    raw = sample()
    raw.loc[4, "最高"] = 10.2
    with pytest.raises(ValueError, match="high is inconsistent"):
        normalize_15m_bars(raw, symbol="600000", captured_at="2026-11-13T03:00:00Z")
