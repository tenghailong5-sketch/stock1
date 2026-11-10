import pandas as pd
import pytest

from src.providers.cnequity import load_daily_bars, normalize_daily_bars


def sample():
    return pd.DataFrame([{
        "symbol": "600000.SH", "trade_date": "2026-11-10",
        "open": 10.0, "high": 10.5, "low": 9.8, "close": 10.2,
        "volume": 10000, "amount": 102000.0, "source": "test",
        "data_version": "v2", "fetched_at": "2026-11-10T08:00:00Z",
    }])


def test_normalize_daily_bars_keeps_retrieval_time_out_of_pit():
    result = normalize_daily_bars(sample())

    assert result.loc[0, "historical_pit_eligible"] is False
    assert pd.isna(result.loc[0, "available_time"])
    assert result.loc[0, "pit_exclusion_reason"] == (
        "retrieval_time_is_not_historical_availability"
    )


def test_rejects_legacy_volume_unit_version():
    frame = sample()
    frame.loc[0, "data_version"] = "v1"

    with pytest.raises(ValueError, match="data_version=v2"):
        normalize_daily_bars(frame)


def test_rejects_duplicate_symbol_date():
    frame = pd.concat([sample(), sample()], ignore_index=True)

    with pytest.raises(ValueError, match="duplicate"):
        normalize_daily_bars(frame)


def test_rejects_invalid_ohlc():
    frame = sample()
    frame.loc[0, "high"] = 9.0

    with pytest.raises(ValueError, match="invalid high"):
        normalize_daily_bars(frame)


def test_load_requires_bounded_symbols_and_uses_dataset_contract():
    calls = []

    def fake_loader(dataset, **kwargs):
        calls.append((dataset, kwargs))
        return sample()

    result = load_daily_bars(["600000.SH"], loader=fake_loader)

    assert calls == [("daily_bars", {"symbols": ["600000.SH"]})]
    assert len(result) == 1


def test_load_rejects_empty_symbols():
    with pytest.raises(ValueError, match="at least one"):
        load_daily_bars([])
