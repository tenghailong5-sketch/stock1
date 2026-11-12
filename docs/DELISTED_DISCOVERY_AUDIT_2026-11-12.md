# Delisted discovery and identity cross-check — 2026-11-12

## Evidence run

- Repository: `tenghailong5-sketch/stock1`
- Main-branch run: `43149026053`
- Run commit: `fa26e1d6feb8d852f6359a6f1fbb63feaafea7b1`
- Baostock observation time: `2026-11-12T03:45:09.015740+00:00`
- Artifact: `delisted-discovery-evidence` (artifact ID `13913666119`)

## Bounded Sina discovery result

| Measure | Result |
|---|---:|
| Probe slots requested | 500 |
| Probes completed | 499 |
| Positive Sina history responses | 421 |
| No-history responses | 78 |
| Failed probes | 1 |
| Remaining code probes | 35,251 |

The one failed probe was `600410.SH` (server disconnected without a response). The upstream discovery routine leaves failed probes pending; it does not mark them as never-issued.

**A positive Sina history response is not proof of formal delisting.** It only proves that the endpoint returned historical bars for the code.

## Baostock identity cross-check

The free `baostock.query_stock_basic` response contained 7,291 rows. The audit selected 5,565 SH/SZ stock rows:

| SH/SZ stock identity | Count |
|---|---:|
| Formal delisted according to Baostock | 341 |
| Currently listed according to Baostock | 5,224 |

Cross-checking the 421 positive Sina probe hits:

| Classification | Count | Share of positive probes |
|---|---:|---:|
| Matched Baostock formal delistings | 63 | 15.0% |
| Matched currently listed stocks | 358 | 85.0% |
| Not present in this Baostock snapshot | 0 | 0.0% |

This demonstrates why raw probe hits must not be treated as a delisted-stock universe. In this batch, 358 of 421 positive hits were still listed according to the current Baostock snapshot.

Baostock's adapter documents SH/SZ coverage and does not claim Beijing-market coverage. The cross-check is therefore not an all-market identity certification.

## Checkpoint/cache note

The restore step for run `43149026053` logged:

```text
Cache not found for input keys: delisted-catalog-main-43149026053, delisted-catalog-main-
```

The cache path list had changed when the identity-evidence file was added alongside the catalogue, changing the Actions cache version. Consequently this run did **not** continue the earlier cache's probe cursor; it re-probed a bounded first chunk. At job completion, Actions saved the new cache under key `delisted-catalog-main-43149026053` successfully.

The next main-branch run verified the new cache version and continued the cursor successfully.

## Incremental checkpoint verification — run 43150217437

- Main-branch commit: `0aa7b193545fc6361907d46bf9fbffcef0b9cbbf`
- Artifact: `delisted-discovery-evidence` (artifact ID `13913059885`)
- Cache restore: `Cache restored from key: delisted-catalog-main-43149026053`
- Cache save: `Cache saved with key: delisted-catalog-main-43150217437`

The log begins with `600410.SH`, the single failed symbol from the prior run, and records it as a successful retry. This confirms the checkpoint resumed instead of restarting at the beginning of the code space.

### Second bounded batch and cumulative discovery state

| Measure | Second run | Cumulative checkpoint |
|---|---:|---:|
| Probe slots | 500 | 1,000 |
| Completed probes | 500 | 999 |
| Positive Sina history responses | 453 | 874 |
| No-history responses | 47 | 125 |
| Failed probes remaining | 0 | 0 |
| Pending code probes | 34,751 | 34,751 |

The cumulative total is the persisted catalogue after the second run. The earlier pre-schema-change run is not added again because the cache-version miss caused the first post-change run to repeat that bounded chunk.

### Updated Baostock cross-check

The same current Baostock snapshot (7,291 rows; 5,565 SH/SZ stocks; 341 formal delistings; 5,224 currently listed stocks) was compared with all 874 cumulative positive Sina hits:

| Classification | Count | Share of positive probes |
|---|---:|---:|
| Matched Baostock formal delistings | 128 | 14.6% |
| Matched currently listed stocks | 746 | 85.4% |
| Not present in this Baostock snapshot | 0 | 0.0% |

The 128 formal matches are still only a subset of Baostock's 341 formal SH/SZ delistings. Discovery remains incomplete. The 746 current-listing matches again show that a positive Sina history response cannot be treated as a delisting label.

## Eligibility and model impact

- The Baostock identity snapshot is a current vendor observation, not historical publication-time evidence.
- `historical_pit_eligible=false` remains in both discovery and identity evidence.
- This audit does not certify daily-bar continuity, corporate-action completeness, delisted-universe completeness, or survivorship-free backtest eligibility.
- No FHM-100 factor weights, hard gates, 80-point entry threshold, or trading rules were changed.
- No live FHM-100 stock scan was executed by this discovery run.
