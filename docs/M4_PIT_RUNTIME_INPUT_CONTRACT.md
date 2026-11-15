# M4 PIT runtime input contract

The live M4 bridge reads four explicitly certified archives from
`FHM_M4_PIT_INPUT_DIR`. Each input may be CSV or Parquet, with the same stem:

- `stock_bars.csv|parquet`
- `stock_returns.csv|parquet`
- `industry_returns.csv|parquet`
- `market_returns.csv|parquet`

Set `FHM_M4_EXPECTED_UNIVERSE_SIZE` to the exact declared point-in-time
universe size. `FHM_M4_PIT_COVERAGE_FLOOR` defaults to 0.80.

## Common provenance gate

All four frames must carry:

- `historical_pit_eligible=true` on every row
- `availability_basis` equal to `original_publication_time` or
  `exchange_publication_time`
- timezone-aware `available_time`, no later than the decision cutoff
- non-empty `source_id` and `source_revision`

A retrieval timestamp, a current membership snapshot, a present-day vendor
download of historical prices, or a source hash generated at retrieval time
does not satisfy this contract.

## Data fields

| Input | Required analytical fields |
|---|---|
| stock_bars | `symbol, trade_date, open, high, low, close, volume, amount` |
| stock_returns | `symbol, trade_date, industry_code, pit_total_return_return_5, pit_total_return_daily_return, pit_total_return_available_time, corporate_action_status` |
| industry_returns | `trade_date, industry_code, return_5, return_20, industry_return, industry_return_5` |
| market_returns | `trade_date, market_return, market_return_5, market_return_20` |

The stock-return archive must be based on a verified point-in-time total-return
series. Corporate-action completeness is a separate requirement; unknown
coverage is not equivalent to no corporate actions. Industry returns must use
the industry classification actually valid for each historical session, with
its original publication/availability lineage. Do not map old sessions using
today's constituents.

The builder emits one decision-time cross-section. Historical replay must call
it separately at each historical decision instant; it must not compute a
single current snapshot and reuse it for older dates.

## Current repository evidence (2026-11-14 archive report)

The existing forward archive is useful for forward observation and research,
but it is not a certified PIT input set:

- Industry membership coverage is 5,579 / 5,579 symbols, but the snapshot is
  explicitly current/retrieval-time only.
- The SW effective-date ledger has 12,934 rows, but its original publication
  availability is not established; it is marked PIT-ineligible.
- SW industry-index history archived 0 of 337 requested codes.
- Stock daily archives contain only 60 per-symbol files in the repository
  tree, and their manifests mark historical rows `retrieval_time_only`.

Therefore the existing archive must not be silently converted into the four
M4 inputs. The runtime loader reports missing or invalid certified inputs and
keeps M4 factors unavailable until a separately sourced, lineage-complete PIT
archive is supplied.
