# CNEquity bounded validation report — 2026-11-11 (updated)

## Latest execution evidence

Two independent pull-request validations completed successfully against the same pinned CNEquity workflow:

| Validation | Workflow run | Result |
|---|---:|---|
| Residual-strength formula PR #14 | [43014556856](https://github.com/tenghailong5-sketch/stock1/actions/runs/43014556856) | success |
| Corporate-action coverage PR #15 | [43015807839](https://github.com/tenghailong5-sketch/stock1/actions/runs/43015807839) | success |

Both runs uploaded a `cnequity-demo-evidence` artifact (14-day retention):
- [PR #14 evidence artifact](https://github.com/tenghailong5-sketch/stock1/actions/runs/43014556856/artifacts/13852637583)
- [PR #15 evidence artifact](https://github.com/tenghailong5-sketch/stock1/actions/runs/43015807839/artifacts/13852863801)

The main-branch merge of PR #14 is `a2105f26ccc7386db5e9d899dd2bf9ef40ba966e`; PR #15 is `b4383b041a9c73eb5a32a6fc08f65ce7dcc7ec14`.

## Bounded current-symbol daily bars and adjustment factors

- 20 explicitly requested A-share symbols.
- 90 sessions per symbol; 1,800 daily-bar rows total.
- Trade-date window: 2026-07-01 through 2026-11-11.
- Missing requested symbols: 0.
- Source: `tdx_protocol`; schema version: `v2`.
- Volume contract: integer shares; non-integer volume rows: 0.
- HFQ adjustment-factor rows: 1,800 from source `sina`.
- Matched HFQ factor keys: 1,800/1,800 (100% sample coverage).
- Factors were finite, positive and unique by symbol/date/adjustment type.
- Strict HFQ query returned the same 1,800 rows with finite adjusted close values.

This validates a bounded recent sample, not the entire A-share market or a long historical interval.

## Corporate-action sample

- Requested symbols: 20.
- Returned records: 115.
- Symbols with returned events: 19/20.
- Observed event types: `bonus`, `cash_dividend`.
- Reported source: `tdx_protocol`.
- Ex-date range: 2023-04-18 through 2026-11-09.
- CNEquity synchronized adjustment factors for 19 affected symbols: 19 processed, 19 updated, 0 skipped, 0 failed, 0 lagging; 1,800 factor rows written.

**Critical limitation:** provider result explicitly reports `coverage_status=unknown`. One selected symbol had no returned event rows, which is not proof that it never had corporate actions. This is evidence that the pipeline can retrieve and synchronize a bounded sample, not proof of a complete issuer corporate-action ledger. The corporate-actions dataset was also reported as `STALE` by the lake status check. No PIT eligibility is established.

## Delisted-security recovery sample

- The bounded discovery/backfill workflow identified 48 catalog entries for the delisted profile.
- Daily-bar backfill wrote 74,486 rows for 48 symbols.
- Requested history start: 2016-01-01.
- Changed month partitions: 2016-01 through 2026-06.
- CNEquity corrected its daily-bar watermark to 2026-04-29 because the prior watermark was ahead of complete data.
- The workflow emitted calendar coverage warnings and rebuilt the trading calendar from seed and bar evidence.

This is a successful bounded recovery run, not a certified complete survivorship-free delisted universe. The run's 48-symbol sample is not proof that every delisted issuer has been discovered or recovered. Recovered bars are not historical PIT-certified.

## Full-lake quality audit — interpret carefully

The enforced `cne audit --full --quality-only` quality gate completed successfully. The audit reported:

- 0 quality errors.
- 1 warning and 4 informational findings.
- 45 registered datasets empty in this demo lake (expected for the bounded demo profile).
- `corporate_actions` marked `STALE`.
- A `trading_status` warning: 1,800 keys were excluded from the count because listing dates were unknown.
- The lake status text reported `UNHEALTHY` overall, while the quality gate itself reported `OK` because freshness was assessed separately.

Therefore, do not summarize this as “the whole lake is healthy” or “all data is complete.” The quality gate passing means the enforced quality checks did not find errors; it does not clear freshness, coverage, or missing-dataset limitations.

## What remains unavailable or unproven

The sample did not establish production-ready historical coverage for:
- complete historical industry membership and index constituents;
- full-market historical breadth and survivorship-free universe membership;
- historical trading status, suspensions and limit-price/limit-event histories;
- complete corporate-action event coverage and payment lifecycle;
- announcements/news with first-publication timestamps;
- fund-flow and minute-level limit-up/down event histories.

Daily bars and corporate actions retain retrieval-time provenance. `fetched_at` is not proof of when the information first became publicly available. These datasets remain `historical_pit_eligible=false`.

## Decision

- Accept CNEquity as a candidate for bounded recent daily-bar ingestion, sample HFQ-factor integrity checks, bounded corporate-action retrieval, and a limited delisted-history recovery experiment.
- Do not connect these retrieved rows to historical FHM-100 backtests or claim PIT-safe performance.
- Do not treat the corporate-action sample as complete while `coverage_status=unknown` and the dataset is stale.
- Do not infer complete delisted-universe coverage from 48 recovered symbols.
- Keep formal historical scoring/backtest admission blocked until historical availability, universe membership, corporate actions, trading status, suspensions and execution evidence are independently verified.
- Do not lower FHM-100 score floors, impute missing factors, or manufacture A/B/C signals.

