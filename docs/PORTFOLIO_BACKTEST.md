# FHM-100 portfolio event simulator

Entry point: `src/model/portfolio_backtest.py::run_portfolio_backtest`.

## Event ordering

1. Refuse the dataset unless `require_backtest_ready(candidates, bars)` passes.
2. Treat an eligible close signal on decision date D as an order for the next available market session, never as a same-close fill.
3. At each session, evaluate protective stops only for positions carried into that session. A stop is triggered when the session low reaches or crosses the stop. For a gap below the stop, the simulator uses the lower session open as the reference price.
4. Route exits and entries through `simulate_daily_fill`, including T+1, explicit suspension/limit-lock flags, volume participation, slippage and fees.
5. Rank same-session entries by score descending, then symbol and strategy for deterministic tie-breaking. Only one signal per symbol can enter; the portfolio limits are enforced by `size_long_position`.
6. Mark remaining positions to the latest available close. Open positions at the end are marked, not force-liquidated.

## Output

The result contains the audit status, initial and ending cash/equity, total return, realized/unrealized P&L, open positions, fills, completed trades, skipped-entry reasons, daily equity observations, and a `metrics` object calculated from the equity curve and completed positions.

## Performance metrics\n\n- Maximum drawdown is measured from the initial-capital high-water mark and then the running peak of daily marked-to-market equity to the subsequent trough. If the drawdown starts from initial capital before the first observation, the peak date is `null` rather than fabricated; the trough date is still reported.\n- Win rate is winning completed positions divided by all completed positions; break-even positions remain in the denominator. Partial exit fills are aggregated by symbol and entry date into one round trip before trade statistics are calculated.\n- Profit factor is gross realized profit divided by absolute gross realized loss. It is `null` when there are no losing records rather than being reported as infinity. Payoff ratio is average win divided by the absolute average loss and is `null` unless both sides exist.\n- Maximum consecutive losses is calculated in exit-date order; a break-even or winning record resets the streak.\n- Empty trade history yields `null` for trade-dependent ratios and averages. Invalid/non-finite equity or P&L values and non-increasing equity dates are rejected.\n\n## Assumptions and limitations

- Candidate rows must include an explicit `stop_price`; the engine will not invent one.
- The event engine is long-only and uses daily bars. Intraday order sequencing, queue priority, partial stop execution paths, corporate-action cash/stock adjustments, dividends, delisting settlement, taxes by historical effective date, and borrow/shorting are not modeled.
- A daily low crossing a stop is a conservative trigger convention, not proof that a stop order was executable at that exact price. Explicit locked-limit and suspension flags remain mandatory.
- The engine uses the sizing module's estimated 30 bps round-trip risk allowance and the execution module's configured cost assumptions. These are research assumptions, not a historical fee schedule.
- No real-world historical performance claim is valid unless independently sourced point-in-time candidate data, complete survivorship-safe universe history, corporate actions and tradability events pass the admission audit.
- The current AkShare retrieval-time snapshots do not meet those requirements. This engine's unit tests validate event logic only; they do not validate strategy profitability.
