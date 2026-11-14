# Kronos integration boundary

## Current implementation

`src/model/kronos_adapter.py` is an optional, lazy-loaded inference adapter for
[shiyu-coder/Kronos](https://github.com/shiyu-coder/Kronos). It uses the upstream
`KronosPredictor.predict(df, x_timestamp, y_timestamp, pred_len, ...)` API and
defaults to the 24.7M-parameter `NeoQuasar/Kronos-small` model with a 512-bar
maximum context. The default lookback is 256 bars and the default device is CPU.

The adapter:

- Requires timestamp, timezone-aware availability time, OHLC, volume and amount.
- Excludes every row not available by the frozen decision time.
- Rejects duplicate bars, malformed OHLC, missing values and non-finite values.
- Requires the caller to supply future exchange-session timestamps; it does not
  invent weekdays or ignore Chinese exchange holidays.
- Emits descriptive forecast-path returns/range/volatility/drawdown only.
- Never manufactures a probability, M5 score, PIT eligibility flag or buy signal.

## Deliberately not wired into the live decision gates yet

The current stock archive labels downloaded history as retrieval-time-only and
not historical point-in-time eligible. A historical K-line fetched today cannot
be treated as information that was available on its original trading date.
Likewise, a current provider snapshot cannot establish historical industry
membership or historical decision-time OHLCV availability.

Therefore this PR adds a tested inference boundary, not a production trading
signal. Do not connect the forecast to M2/M3/M5/M8 authorization until the
upstream A-share OHLCV adapter supplies verified point-in-time availability,
corporate-action-consistent price basis, exchange-session calendars, and
historical coverage. Then run rolling, time-ordered out-of-sample comparisons
against the existing model and simple baselines before considering a calibrated
forecast factor.

## Optional runtime dependencies

The existing scan requirements remain unchanged. Kronos and its PyTorch /
Hugging Face dependencies are loaded only when the adapter is asked to load a
real predictor. Place a pinned upstream Kronos checkout outside the stock1
repository and set `KronosConfig(repo_path="/path/to/Kronos")`; the adapter adds
that directory to the import path only at model-load time. The default weights
are downloaded from Hugging Face on first load, so keep model installation and
weight download out of the normal daily scan until resource use and
reproducibility have been measured.
