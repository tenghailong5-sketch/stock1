# FHM-100 Lite

A small A-share live-quote watchlist scanner.

## What it does

1. Fetches a current A-share quote snapshot with provider fallback: Eastmoney → Tencent → Sina.
2. Removes ST/delisting names and invalid quotes.
3. Keeps stocks with traded value of at least CNY 50 million and daily change between -2% and +7.5%.
4. Ranks the remaining stocks using only observed cross-sectional data:
   - Daily-change percentile: 65%
   - Traded-value percentile: 35%
5. Publishes up to 20 research-watch candidates.

## Outputs

- `output/market_snapshot.csv`: quote snapshot used by this run.
- `output/final_candidate_shortlist.csv`: ranked watchlist.
- `output/simple_scan_audit.json`: provider, coverage, scoring basis and limitations.
- `output/report.md`: readable report.
- `output/formal_gate.json`: run status; buy authorization is always false.

## What it deliberately does not do

This version does **not** claim to identify industry mainlines, policy catalysts, capital concentration, market-emotion turning points, multi-day trends, 15-minute entry confirmation, M2/M3/M5/M8 scoring, account-level position sizing, stop prices, backtests, or profitable trades. The available quote snapshot does not support those conclusions.

A candidate is a watchlist observation, not a buy signal. Missing information is not fabricated.

## Run

Use the **FHM-100 Daily Research** GitHub Actions workflow manually when a scan is wanted. A push to `main` also triggers a run. No scheduled run is configured.

## Test

```bash
pip install -r requirements-simple.txt
python -m pytest -q tests/test_simple_fhm100.py
python -m src.simple_fhm100
```

Historical archive files are retained as raw reference material only; the Lite runtime does not read them.
