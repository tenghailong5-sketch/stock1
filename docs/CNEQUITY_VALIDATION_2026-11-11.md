# CNEquity bounded validation report — 2026-11-11

## Run identity

- Repository: `tenghailong5-sketch/stock1`
- Branch: `feat/fhm100-m5-pipeline`
- Commit: `d5e2841f4944011f3b2cf0b03c7a7c0ca1474221`
- Workflow: CNEquity Bounded Demo Validation
- Run: https://github.com/tenghailong5-sketch/stock1/actions/runs/42931432242
- Result: success

## Verified sample

- 20 explicitly requested A-share symbols across Shanghai/Shenzhen main boards, ChiNext and STAR Market.
- 90 daily sessions per symbol, 1,800 daily-bar rows total.
- Coverage window: 2026-06-30 through 2026-11-10.
- Missing requested symbols: 0.
- Daily-bar volume rows use integer share units under the CNEquity v2 schema.
- 1,800 HFQ adjustment-factor rows from source `sina`; 1,800/1,800 symbol-date keys matched (100% sample coverage).
- Strict HFQ query returned 1,800 rows; adjusted close values were finite.
- Adjustment factors were finite, positive and unique by symbol/date/adjustment type.
- CNEquity full data-lake audit: 0 errors.
- Latest run audit: 0 errors, 0 warnings.

Evidence is retained in the workflow artifact `cnequity-demo-evidence` for 14 days. The artifact includes `data-validation.json`, `adj-factor-validation.json`, `strict-adjustment-check.json`, `check.log`, and `status.log`.

## What this does not establish

The full audit's green verdict is a data-quality verdict for the datasets currently stored; it is not a completeness or historical-PIT certification.

At the time of this run, the lake contained 4 datasets and 6,492 rows. The four populated datasets were:

- `daily_bars`
- `adj_factors`
- `instruments`
- `trading_calendar`

The status report showed 82 other datasets without data. In particular, the run did not establish usable historical coverage for:

- `corporate_actions`
- `delisting_events`
- `trading_status`
- `limit_events` / `limit_prices`
- `industry_members`
- `index_constituents`
- `market_breadth`
- `announcement_index` and `financial_statement_items`

The sample is only 20 selected currently requested symbols and 90 recent sessions. It is not a full-market historical universe, does not include a verified survivorship-free list of delisted securities, and does not validate historical membership or historical trading restrictions.

Daily bars have retrieval-time provenance; `fetched_at` is not proof of when a bar was historically available to a strategy. The adapter therefore correctly retains `historical_pit_eligible=false`. The derived HFQ factor coverage test validates key coverage and numeric integrity, not the completeness or correctness of the underlying corporate-action event ledger.

## Decision

- Accept the CNEquity integration for bounded recent daily-bar ingestion and numeric/coverage validation only.
- Do not connect these observations to historical FHM-100 backtests or claim PIT-safe performance.
- Do not infer delisting, suspensions, limit locks, corporate actions, or historical universe membership from the current instrument list or daily bars.
- Keep the historical backtest admission gate closed until each missing evidence family is populated and independently audited.
- Do not reduce the FHM-100 score floor or impute missing factors to manufacture candidates.
