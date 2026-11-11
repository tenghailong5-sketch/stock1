# A-share execution simulator

## Scope

`src/model/execution.py` provides deterministic daily-bar fill simulation for research and backtesting. It is not a broker adapter, order router, or queue-position model.

## Fill constraints

- Orders use positive integer board-lot quantities.
- Python and NumPy integer scalars are accepted for quantities, lot sizes, and reported volume, then normalized to Python integers before arithmetic.
- Execution-state flags must be explicit booleans. Python `bool` and NumPy `bool_` are accepted; nulls and strings such as `"false"` are rejected.
- Suspended securities do not fill.
- A buy is blocked when the upper-limit queue is explicitly marked locked; a sell is blocked when the lower-limit queue is explicitly marked locked.
- T+1 blocks a sale on or before the entry date.
- Missing or invalid volume fails closed. Maximum participation limits the filled quantity and can produce a partial fill.

## Cost assumptions

The default research assumptions are 2.5 bps commission, 5 currency units minimum commission, 5 bps sell stamp duty, 10 bps slippage, and 0.1 bps transfer fee. These are configurable estimates, not a historical fee schedule. A production backtest must supply date-appropriate fees and independently verified execution-state observations.

## Evidence boundary

Daily OHLCV alone cannot establish whether an order could execute in a locked limit queue. The caller must provide independently sourced suspension and limit-lock flags. The simulator must not infer those flags from daily percentage change.

The simulator does not yet provide a complete portfolio event loop, corporate-action processing, or historical performance metrics. Historical backtests remain blocked until the separate point-in-time data admission audit passes.
