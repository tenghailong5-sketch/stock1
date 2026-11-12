# CNEquity bounded validation: data-readiness findings (2026-11-12)

## Purpose

This note records what the successful GitHub Actions CNEquity bounded validation actually establishes. A successful job means the bounded collection and validation pipeline completed; it does **not** certify historical point-in-time (PIT) eligibility or complete market-wide coverage.

Evidence source: GitHub Actions run `43135020175`, artifact `cnequity-demo-evidence` (created 2026-11-12). The artifact is retained by GitHub Actions and is not copied into this repository.

## Observed evidence

| Dataset | Observed result | What it does not establish |
|---|---:|---|
| Current bounded daily bars | 20 symbols, 2,420 rows; 121 sessions per symbol, 2026-05-18 through 2026-11-11; zero malformed OHLCVA rows | Historical PIT availability; the validation explicitly sets `historical_pit_eligible=false` because retrieval time is not historical availability |
| Main-universe corporate actions | 115 rows across 19 of 20 symbols; event dates 2023-04-18 through 2026-11-09; TDX protocol source | Complete event coverage or historical public-availability timestamps; `historical_pit_eligible=false` |
| Delisted-universe corporate actions | 195 rows across 16 of 17 catalog symbols; event types include allotment, bonus and cash dividend; dates 1997-08-18 through 2026-05-27 | Complete delisted security universe, event completeness, or historical PIT eligibility; explicitly false |
| Main-universe adjustment factors | 2,420 HFQ rows; 121/121 session matches for each of 20 bounded symbols; Sina source | Historical PIT eligibility; explicitly false |
| Delisted daily bars | 15,006 rows reported for 11 symbols | Continuous coverage to the present. Audit corrected a claimed 2026-11-11 watermark: freshest complete curated trade date was 2025-04-10, and the source produced no data for the intervening period |
| Delisted adjustment factors | 15,006 rows synchronized for 11 affected symbols | Completeness of the delisted universe or PIT eligibility |

## Admission decision

1. Keep the successful bounded demo as evidence that these free-source collectors can retrieve and validate a limited sample.
2. Do not set corporate-action coverage to `COMPLETE` based only on a non-empty response, adjustment-factor derivation, or successful workflow.
3. Do not use these backfilled daily bars, corporate actions, or derived adjustment factors in historical PIT backtests unless their historical availability and complete coverage are independently attested.
4. Do not infer a symbol had no corporate actions when the provider returned no rows unless a source-backed complete ledger attestation explicitly covers that symbol and interval.
5. Keep the strict PIT builder fail-closed. A production adapter must bind each event set to a stable ledger snapshot, carry source/revision and availability timestamps, prove coverage for the entire requested interval, and reject events whose ex-dates have no matching price bars.

## Next engineering gates

- Identify a free source or reproducible official source process that can attest the complete event ledger per symbol and interval.
- Distinguish event effective date from first public-availability timestamp; retrieval time alone is not a substitute.
- Improve delisted-universe discovery and quantify missing symbols and missing trading periods before any survivorship-bias claim.
- Integrate the normalized PIT builder into the production factor pipeline only after the provider contract can populate all required fields without fabricating attestations.

No FHM-100 factor weights, A/B/C scoring rules, hard gates, or 80-point entry threshold are changed by this evidence note.
