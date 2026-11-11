# CNEquity bounded validation evidence — 2026-11-11

## Run identity

- GitHub Actions run: [43040704055](https://github.com/tenghailong5-sketch/stock1/actions/runs/43040704055)
- Workflow branch: `fix/cnequity-demo-daily-bars-prerequisite`
- Workflow head: `cd1429e67d6da078aff362dd5cbd817aacd5fd46`
- Result: all 18 substantive workflow steps completed successfully.
- Evidence artifact: `cnequity-demo-evidence` (artifact ID `13866261462`, SHA-256 `cd0b8b5b81785defe344ee7eef7555690d33a226a0c91ee5ca5b91369482904f`).
- PR #18 was merged as `d79a3d2c69c7efd22539206996e62a52bc84f683`.

## Bounded daily-bar sample

- 20 explicitly requested A-share symbols; no requested symbol missing.
- 2,440 daily-bar rows, 122 distinct sessions per symbol.
- Trade-date range: 2026-05-15 through 2026-11-11.
- Source: `tdx_protocol`; schema version `v2`; volume unit: shares.
- All 2,440 rows passed finite OHLCVA validation; invalid rows: 0.
- Volume values were integer-valued for all 2,440 rows.
- This is a bounded sample, not a full-market universe.

## Adjustment factors and corporate actions

- HFQ adjustment-factor rows: 2,440; source: `sina`.
- All factors finite and positive; primary keys unique.
- HFQ factor coverage: 122/122 sessions (100%) for each of the 20 sample symbols.
- Strict HFQ load returned 2,440 rows and finite adjusted closes.
- Corporate-action backfill returned 115 records across 19 of 20 symbols, event types `bonus` and `cash_dividend`, source `tdx_protocol`.
- Corporate-action coverage remains **unknown/incomplete**. The absence of an event for the remaining symbol is not proof that it had no corporate actions.
- Corporate actions were marked **STALE** by the lake audit. The sample sync completed, but the ledger must not be treated as complete.

## Delisted-history recovery experiment

- Discovery sweep probed 500 of 35,731 pending instrument codes: 422 catalog candidates, 78 never-issued, 0 failed; sweep reported `complete: false`, with 35,231 still pending.
- The existing delisted catalog reported 63 records, of which 48 fell within the requested 2016-01-01 onward window and were backfilled.
- Backfill reported 74,486 daily-bar rows for 48 symbols.
- The workflow also extended the trading calendar with 4,333 rows.
- This is a bounded recovery experiment, **not a complete or survivorship-free delisted-security universe**.
- The delisted-history audit corrected a misleading daily-bar watermark: it claimed 2026-11-11 while the freshest complete curated date was 2026-04-29. Source data produced no bars for the intervening dates.
- Trading-status coverage remained unknown. Do not infer that a stale last-traded date alone proves formal delisting.

## Quality audit and limits

The full quality audit reported:

- 0 errors, 1 warning, 4 informational findings.
- Quality gate: **OK**.
- Overall demo-lake health: **UNHEALTHY**, because freshness is assessed separately and `corporate_actions` was stale.
- `trading_status` warning: 2,440 sample keys were excluded from coverage counts because listing dates were unknown.
- 45 registered datasets were empty as expected for this isolated demo profile.
- Research readiness was not assessed for this demo lake.

Most importantly, all historical sample bars and corporate-action rows are **not point-in-time eligible**: retrieval timestamps do not prove when information was first publicly available. This run does not validate historical PIT performance, a historical backtest, full-market breadth, historical industry/index membership, or M5 trading signals.

## Decision

Accept this run as evidence that the bounded real-provider ingestion path, daily-bar validation, sample HFQ alignment, and limited delisted-history recovery can complete end-to-end. Do not mark the entire CNEquity lake healthy or complete. Do not enable M5 scoring or issue formal A/B/C signals from this dataset. Keep the existing fail-closed PIT and factor-completeness requirements unchanged.
