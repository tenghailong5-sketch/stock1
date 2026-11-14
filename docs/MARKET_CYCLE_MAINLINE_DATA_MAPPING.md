# Policy / Emotion / Mainline sub-system data mapping

This is an additive research subsystem. It does not renumber FHM-100 M1-M9:
M3 remains market environment, M4 remains factors, M5 remains A/B/C strategy
scoring, M6 remains return/risk prediction, M7 execution, M8 portfolio risk,
and M9 validation/monitoring.

## Three independent engines

| Engine | Inputs | Output | Source contract |
|---|---|---|---|
| Policy catalyst | official policy/announcement records | visible event + industry mapping | published_at, available_at, source_url, source_id, event type, industry codes |
| Market emotion cycle | breadth, limit-up/down counts, optional board-failure and promotion rates | ICE / RECOVERY / HEAT / EXHAUSTION / TRANSITION | same-session market aggregates with capture and availability timestamps |
| Mainline flows | daily industry return, market return, industry turnover and market turnover | IGNITION / PEAK / DECAY | industry daily series and contemporaneous industry taxonomy |

## Existing data mapping

- Market breadth and turnover: `data/forward_archive/market_breadth/{trade_date}.jsonl`.
- Limit-pool counts: `data/forward_archive/market_sentiment/{trade_date}.jsonl`.
- Index daily bars: `data/forward_archive/index_daily/{index_symbol}.jsonl`.
- Industry membership observations: `data/forward_archive/industry_membership/`; retrieval-time observations are forward-only.
- Existing official SW2021 latest snapshot remains forward-only. It must not be backfilled across historical dates.
- Policy event archive is a new required input; until source-backed event records are collected, the policy engine reports no visible events rather than synthesizing them.
- Historical industry index/turnover series and historical constituent membership with original effective/publication timestamps are required for a valid historical mainline test.

## Time and quality gates

Every engine distinguishes event/session time from system availability time. A row may be used at decision time only when both its publication/capture and availability timestamps are no later than the decision cutoff. Missing required observations remain missing; no zero fill.

The current forward archive is not retroactively PIT eligible. All outputs in this first implementation carry `historical_pit_eligible=false`. Historical replay must use a separately verified vendor history with original publication/availability semantics and survivorship-safe universe.

## Lifecycle research protocol

1. Run the emotion engine on daily full-market breadth and limit-pool series.
2. Join policy catalysts to industries only after publication/availability checks.
3. Run mainline classification on contemporaneous daily industry return and turnover share.
4. Compare inferred IGNITION / PEAK / DECAY dates against manually documented historical episodes, using frozen rules and no future data.
5. Report event precision/recall, lead/lag, and false alarms; split by bull, bear, and range regimes.
6. Do not connect these descriptive labels to FHM candidate scores until out-of-sample tests pass.

The current code provides deterministic classification contracts and unit tests only. It does not claim historical market episodes have been validated.
