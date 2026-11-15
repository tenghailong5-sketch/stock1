# FHM-100 portfolio event simulator

Entry point: `src/model/portfolio_backtest.py::run_portfolio_backtest`.

## Event ordering

1. Refuse the dataset unless `require_backtest_ready(candidates, bars)` passes.
2. Treat an eligible close signal on decision date D as an order for the next available market session, never as a same-close fill.
3. At each session, evaluate protective stops only for positions carried into that session. A stop is triggered when the session low reaches or crosses the stop. For a gap below the stop, the simulator uses the lower session open as the reference price.
4. Detect daily-low stop breaches only after the session closes and execute pending exits no earlier than the next session's open. Opening fills use separately sourced opening-auction matched volume with an availability timestamp no later than 09:30 Asia/Shanghai.
5. Rank same-session entries by score descending, then symbol and strategy for deterministic tie-breaking. Only one signal per symbol can enter. Position sizing consumes the candidate's historical M8 `available_risk_budget` and M3 `market_risk_scale`; there is no fixed per-trade currency cap, portfolio currency cap, or maximum position count.
6. Mark remaining positions to the latest available close. Open positions at the end are marked, not force-liquidated.

## Output

The result contains the audit status, initial and ending cash/equity, total return, realized/unrealized P&L, open positions, fills, completed trades, skipped-entry reasons, daily equity observations, and a `metrics` object calculated from the equity curve and completed positions.

## Performance metrics\n\n- Maximum drawdown is measured from the initial-capital high-water mark and then the running peak of daily marked-to-market equity to the subsequent trough. If the drawdown starts from initial capital before the first observation, the peak date is `null` rather than fabricated; the trough date is still reported.\n- Win rate is winning completed positions divided by all completed positions; break-even positions remain in the denominator. Partial exit fills are aggregated by symbol and entry date into one round trip before trade statistics are calculated.\n- Profit factor is gross realized profit divided by absolute gross realized loss. It is `null` when there are no losing records rather than being reported as infinity. Payoff ratio is average win divided by the absolute average loss and is `null` unless both sides exist.\n- Maximum consecutive losses is calculated in exit-date order; a break-even or winning record resets the streak.\n- Empty trade history yields `null` for trade-dependent ratios and averages. Invalid/non-finite equity or P&L values and non-increasing equity dates are rejected.\n\n## Assumptions and limitations

- Candidate rows must include explicit `stop_price`, `available_risk_budget`, `market_risk_scale`, and boolean `m8_trade_allowed`; the engine will not invent them.
- `available_risk_budget` means M8's remaining risk capacity at that historical decision time, already net of then-current portfolio exposure and other portfolio constraints. `market_risk_scale` is the contemporaneous continuous M3 multiplier (0..1); the simulator applies it exactly once. Existing-position risk is not subtracted from the remaining budget a second time. All three risk fields and the permission flag are mandatory and audited before simulation. An M5-eligible candidate cannot enter if its historical M8 permission is false.
- The backtest uses each candidate's own historical risk context, not a static global substitute. If the historical risk context is unavailable, the PIT backtest is rejected rather than assigning an arbitrary budget.
- The event engine is long-only and uses daily bars. Intraday order sequencing, queue priority, partial stop execution paths, corporate-action cash/stock adjustments, dividends, delisting settlement, taxes by historical effective date, and borrow/shorting are not modeled.
- A daily low crossing a stop is detected only after that session closes; execution is deferred to the next session's open. Opening fills use point-in-time opening-auction matched volume, never same-day total volume. Explicit locked-limit and suspension flags remain mandatory.
- The engine uses the sizing module's estimated 30 bps round-trip risk allowance and the execution module's configured cost assumptions. These are research assumptions, not a historical fee schedule.
- No real-world historical performance claim is valid unless independently sourced point-in-time candidate data, complete survivorship-safe universe history, corporate actions and tradability events pass the admission audit.
- The current AkShare retrieval-time snapshots do not meet those requirements. This engine's unit tests validate event logic only; they do not validate strategy profitability.


### Risk-budget cost treatment
New position sizing reserves an additional fixed 10 currency units for round-trip minimum commissions, on top of its conservative 30 bps variable-cost allowance. Existing-position risk is recalculated from the actual recorded buy-side fees and modeled stop-exit slippage, sell commission (including the minimum), stamp duty, and transfer fee. This is a conservative estimate, not a guarantee of realized maximum loss; gaps, partial fills, suspensions, and locked limit-down exits can still exceed the planned stop risk.

## Pending protective exits and partial fills

- Once a protective stop is triggered, the position remains marked as `stop_exit_pending` until all remaining shares are sold. If a suspension, locked limit-down state, or participation cap prevents a full exit, the simulator retries the pending market exit on subsequent sessions; a rebound does not silently cancel the exit.
- Each partial fill realizes its own proceeds and costs, with the original buy-side cost allocated pro rata to the shares sold. Remaining shares and their remaining allocated buy cost stay in the open position.
- Partial realized P&L is included in portfolio cash/equity accounting, but a position with any shares still open is excluded from completed-trade win-rate, profit-factor, payoff, expectancy, and loss-streak statistics. It becomes one completed round trip only after its remaining shares are fully closed.
- The pending-exit state is exposed on each open-position record as `stop_exit_pending`.

