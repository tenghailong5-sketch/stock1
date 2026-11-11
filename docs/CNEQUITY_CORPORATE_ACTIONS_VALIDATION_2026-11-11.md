# CNEquity bounded corporate-action validation — 2026-11-11

## Execution evidence

- Workflow run: https://github.com/tenghailong5-sketch/stock1/actions/runs/42931829733
- Commit under test: `26b83136b2d72b6889b0f32d12b2640511abd62b`
- CNEquity version: `0.23.0`
- Scope: 20 selected A-share symbols, 2023-01-01 through 2026-11-10.
- Backfill plan was generated before execution.
- Corporate-action rows written: 115.
- Symbols with returned events: 19 of 20.
- Event types observed: `bonus`, `cash_dividend`.
- Source reported: `tdx_protocol`.
- Event ex-date range: 2023-04-18 through 2026-11-09.
- The remaining selected symbol had no corporate-action rows in this returned sample; this is not evidence that no action ever existed.

## Derived-factor consistency

After the corporate-action backfill, CNEquity automatically recomputed adjustment factors for 19 affected securities:

- Processed: 19
- Updated: 19
- Skipped: 0
- Failed: 0
- Lagging: 0
- Recomputed factor rows written: 1,800
- Provider reported corporate actions and adjustment factors synchronized.

The separate bounded daily-bar check still returned 1,800 rows (20 symbols × 90 sessions) with no missing requested symbols. HFQ factors matched all 1,800 sample symbol/session keys.

## Whole-lake quality

- `cne check --full`: 0 errors.
- Latest run audit: 0 errors, 0 warnings.
- 5 datasets populated, 6,607 rows.
- 81 datasets still had no data in this isolated demo lake.

## Limits

- Corporate-action backfill result reports `coverage_status=unknown`; 115 rows and a successful execution do not prove complete event coverage over the requested interval.
- Only the `tdx_protocol` source was reported for this run; the plan listed backup/supplementary sources but the successful result does not prove all sources independently reconciled.
- Events and adjustment factors remain `historical_pit_eligible=false`. Fetch time is not proof of when information was publicly available.
- This does not certify a complete issuer corporate-action ledger, cash-payment lifecycle, rights issue coverage, delisted issuer history, or the historical point-in-time state of the FHM-100 strategy.

## Decision

Accept as evidence that the free CNEquity pipeline can retrieve a bounded sample of corporate actions and automatically synchronize derived HFQ factors for affected sample symbols. Do not treat it as a complete corporate-action ledger or unlock historical performance backtests.
