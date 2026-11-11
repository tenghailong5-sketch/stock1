# CNEquity data-source adoption gate

## Purpose

CNEquity is being evaluated as a persisted A-share data source for FHM-100.
Its public project documentation describes local Parquet storage, incremental
updates, historical instruments/delisted securities, corporate actions,
adjustment factors, and PIT financial-statement availability. These are
capabilities claimed by the upstream project, not yet independently verified
for this repository's deployment.

Upstream:
- https://github.com/rootSunc/CNEquity
- https://rootsunc.github.io/CNEquity/guides/free-a-share-data/
- https://rootsunc.github.io/CNEquity/datasets/sources/

## Integration status

- Optional adapter: `src/providers/cnequity.py`.
- Loads only an explicitly supplied symbol list through
  `cnequity.query.load("daily_bars", symbols=[...])`.
- Requires the documented daily-bar fields and `data_version=v2`, whose
  upstream schema defines volume in shares.
- Rejects missing/non-finite OHLCVA, invalid OHLC, duplicate symbol/date keys,
  missing source lineage, and invalid retrieval timestamps.
- Preserves `fetched_at` as retrieval time.
- Deliberately sets `available_time=NaT` and
  `historical_pit_eligible=False`. Retrieval time is not proof of historical
  public availability.
- The adapter is not yet connected to the live scanner or M5 scoring.

## Acceptance gates before production use

1. Pin and test a specific CNEquity release in a clean GitHub Actions runner.
2. Run the upstream demo profile and a bounded real-symbol smoke test.
3. Run `cne check`; archive its quality report and dataset coverage.
4. Verify daily-bar volume units, symbol suffixes, trading calendar, corporate
   actions, adjustment-factor coverage, and delisted-symbol retention.
5. Verify PIT semantics separately for every factor family. Financial
   statement publication dates do not automatically establish PIT for
   valuation, industry membership, fund flow, news, or sentiment.
6. Map only fields whose definitions and timestamps match FHM-100 factor
   contracts. Missing or stale factors remain hard failures; no imputation.
7. Keep the historical backtest blocked until candidate availability,
   historical universe membership, corporate actions, delisted outcomes,
   suspensions, and limit-lock execution evidence are independently verified.

## Important limitation

CNEquity is a data lake, not the FHM-100 scoring engine or an execution
simulator. The existence of a dataset or a PIT-related field does not itself
prove the entire dataset is complete, correct, or point-in-time safe.
