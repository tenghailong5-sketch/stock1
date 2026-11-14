# Policy / Emotion / Mainline sub-system data mapping

This is an additive research subsystem. It does not renumber FHM-100 M1-M9:
M3 remains market environment, M4 remains factors, M5 remains A/B/C strategy
scoring, M6 remains return/risk prediction, M7 execution, M8 portfolio risk,
and M9 validation/monitoring.

## Three independent engines

| Engine | Inputs | Output | Source contract |
|---|---|---|---|
| Policy catalyst | official policy/announcement records | visible event + industry mapping | published_at, available_at, source_url, source_id, event type, industry codes, impact_direction, confidence, confidence_basis |
| Market emotion cycle | breadth, limit-up/down counts, optional board-failure and promotion rates | ICE / RECOVERY / HEAT / EXHAUSTION / TRANSITION | same-session market aggregates with capture and availability timestamps |
| Mainline flows | daily industry return, market return, industry turnover and market turnover | IGNITION / PEAK / DECAY | industry daily series and contemporaneous industry taxonomy |

## Existing data mapping

- Market breadth and turnover: `data/forward_archive/market_breadth/{trade_date}.jsonl`.
- Limit-pool counts: `data/forward_archive/market_sentiment/{trade_date}.jsonl`.
- Index daily bars: `data/forward_archive/index_daily/{index_symbol}.jsonl`.
- Industry membership observations: `data/forward_archive/industry_membership/`; retrieval-time observations are forward-only.
- Existing official SW2021 latest membership snapshot remains forward-only. It must not be backfilled across historical dates.
- SW first-level index codes and names are retrieved from the official `index_publish/current` API; historical daily close and turnover come from the official `index_publish/trend` API. The adapter keeps TLS verification enabled with a pinned CA fallback, stamps retrieval time, and hashes source rows. Original historical availability is not supplied, so replay remains retrospective/non-PIT.
- `data/policy_catalysts/official_policy_events.jsonl` contains four official policy notices from the Ministry of Ecology and Environment, Ministry of Science and Technology, Ministry of Commerce, and National Data Administration. Sector mappings use the 31-code SW first-level index list retrieved from the official SW API.
- Each policy event requires controlled impact_direction, mapping confidence, confidence_basis, and availability_basis. Date-only sources are conservatively made available at end of publication day; a ministry page with a later web timestamp is not exposed before that page timestamp.
- These four records validate event visibility and sector mapping contracts; they are a seed set, not a complete historical policy corpus or evidence of predictive alpha.
- `scripts/replay_sw_mainlines.py` performs a retrospective official-SW industry-index replay using complete first-level index coverage, 5-day relative strength, and industry turnover share. It fails closed if any current first-level industry index history is unavailable.
- Historical constituent membership with original effective/publication timestamps is still required before any PIT stock-level mainline backtest or formal signal.

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

The mainline lifecycle uses two consecutive qualifying sessions to confirm each state transition; the transition is stamped on the confirmation session and never backdated. This avoids single-day phase flicker and look-ahead leakage. The live replay produces phase segments for inspection. These are retrospective labels, not an out-of-sample predictive validation.
