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

The cache path list must remain stable for subsequent runs to restore this new checkpoint version. The next main-branch discovery run is the verification point: it must log a successful cache restore before advancing to the next chunk.

## Eligibility and model impact

- The Baostock identity snapshot is a current vendor observation, not historical publication-time evidence.
- `historical_pit_eligible=false` remains in both discovery and identity evidence.
- This audit does not certify daily-bar continuity, corporate-action completeness, delisted-universe completeness, or survivorship-free backtest eligibility.
- No FHM-100 factor weights, hard gates, 80-point entry threshold, or trading rules were changed.
- No live FHM-100 stock scan was executed by this discovery run.
