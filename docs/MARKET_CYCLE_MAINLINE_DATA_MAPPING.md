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
- `assemble_market_emotion_history` reads these raw daily archives directly, with the same after-close, source-priority, availability-time, and coverage checks as M3. It does not inherit M3's MA60/volatility warm-up truncation. The replay supplements the archive with AkShare/Eastmoney `stock_zt_pool_em`, `stock_zt_pool_zbgc_em`, and `stock_zt_pool_strong_em` for the latest 15 trading sessions only. It derives highest limit-up streak, strong-pool count, board-failure rate, and first-board promotion rate; missing endpoints remain missing. These fetched historical pools are stamped at retrieval time and are explicitly non-PIT. Older dates are outside the provider's supported rolling window. Each classified session carries `market_cycle_data_basis` so advanced board metrics are not mistaken for full-period coverage.
- Index daily bars: `data/forward_archive/index_daily/{index_symbol}.jsonl`.
- Industry membership observations: `data/forward_archive/industry_membership/`; retrieval-time observations are forward-only.
- Existing official SW2021 latest membership snapshot remains forward-only. It must not be backfilled across historical dates.
- SW first-level index codes and names are retrieved from the official `index_publish/current` API; historical daily close and turnover come from the official `index_publish/trend` API. The adapter keeps TLS verification enabled with a pinned CA fallback, stamps retrieval time, and hashes source rows. Original historical availability is not supplied, so replay remains retrospective/non-PIT.
- `data/policy_catalysts/official_policy_events.jsonl` contains four official policy notices from the Ministry of Ecology and Environment, Ministry of Science and Technology, Ministry of Commerce, and National Data Administration. Sector mappings use the 31-code SW first-level index list retrieved from the official SW API.
- Each policy event requires controlled impact_direction, mapping confidence, confidence_basis, and availability_basis. Date-only sources are conservatively made available at end of publication day; a ministry page with a later web timestamp is not exposed before that page timestamp.
- Policy catalysts are assigned to the first A-share trading-session close (15:00 Asia/Shanghai) at or after `available_at`. Weekend, holiday, and post-close notices therefore become usable on the next eligible session; they are not backdated into a prior close. The five-session catalyst window is measured in trading sessions, not calendar days.
- These four records validate event visibility and sector mapping contracts; they are a seed set, not a complete historical policy corpus or evidence of predictive alpha.
- `scripts/replay_sw_mainlines.py` performs a retrospective official-SW industry-index replay using complete first-level index coverage, 3-day relative-strength acceleration for ignition, 5-day relative strength for lifecycle context, and industry turnover share. The turnover-share denominator is the sum of the 31 SW first-level index amounts on that date, not a separately sourced exchange-wide turnover field. It fails closed if any current first-level industry index history is unavailable.
- `src/model/mainline_validation.py` evaluates phase-transition events against subsequent 5/10/20-session industry excess returns and compares ignition-threshold candidates at 5- and 10-session horizons with chronological 60/40 train/test splits. Forward returns are outcome labels only and are never inputs to phase detection or live scoring. This is retrospective outcome validation, not a PIT backtest.
- Historical constituent membership with original effective/publication timestamps is still required before any PIT stock-level mainline backtest or formal signal.

## Time and quality gates

Every engine distinguishes event/session time from system availability time. A row may be used at decision time only when both its publication/capture and availability timestamps are no later than the decision cutoff. Missing required observations remain missing; no zero fill.

The current forward archive is not retroactively PIT eligible. All outputs in this first implementation carry `historical_pit_eligible=false`. Historical replay must use a separately verified vendor history with original publication/availability semantics and survivorship-safe universe.

## Lifecycle research protocol

1. Run the emotion engine on daily full-market breadth and limit-pool series.
2. Join policy catalysts to industries only after publication/availability checks.
3. Run mainline classification on contemporaneous daily industry return and turnover share.
4. Compare inferred IGNITION / PEAK / DECAY dates against manually documented historical episodes, using frozen rules and no future data.
5. Evaluate transition dates against subsequent 5/10/20-session excess returns as an outcome-only check; separately report event precision/recall, lead/lag, and false alarms against an independently adjudicated episode set.
6. Split results by bull, bear, and range regimes.
7. Do not connect these descriptive labels to FHM candidate scores until out-of-sample tests pass.

The mainline lifecycle uses two consecutive qualifying sessions to confirm each state transition; the transition is stamped on the confirmation session and never backdated. Ignition and peak also require above-median industry turnover-share rank. The replay emits phase segments and a policy-to-subsequent-ignition temporal alignment table. Alignment is descriptive coincidence only, not causality or predictive validation.
