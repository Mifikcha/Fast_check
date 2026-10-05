# Implementation plan: microstructure alpha feasibility probe

Plan frozen before implementation on 2026-09-10. The reference repository was
read at commit `933ddbb57b5a987e3f0313e0483611d04d915bbc`; current Bybit protocol
details and primary-source links are in `reference_research.md`.

## 1. Reused patterns and deliberate departures

Reuse, in reduced form:

- timezone-aware UTC parsing, stable numeric types, chronological sorting and
  explicit deduplication;
- immutable, bounded Parquet parts and a centralized symbol-aware path policy;
- causal `merge_asof(..., direction="backward")` with an added maximum-age
  check;
- trailing (never centered) rolling calculations and explicit numerical guards;
- fail-fast structural sanity checks;
- expanding-window folds, train-only preprocessing, semantic feature groups,
  held-out permutation importance, group ablation, PNG plots, and both human-
  and machine-readable outputs.

Implement from scratch:

- raw public WebSocket capture, transport health and reconnect state;
- a strict snapshot/delta order-book state machine and event-level OFI;
- a complete one-second availability-time grid, trade-flow aggregation, exact
  timestamp targets, and stale/invalid interval propagation;
- outer 60/20/20 train/validation/test isolation plus purge and embargo;
- regression/classification baselines, threshold diagnostics, time-series-aware
  uncertainty, FDR, concentration/regime analysis, verdict logic, and the single
  CLI workflow.

Do not copy the old project's silent overwrites, global zero filling, unlimited
forward fill, `bfill`, hard-coded fees/funding clock, row shifts after dropping
gaps, warning-only gap handling, or use of the full tail during feature/model
selection.

## 2. Current Bybit sources

Use the unauthenticated linear public endpoint
`wss://stream.bybit.com/v5/public/linear` and subscribe to:

- `orderbook.50.BTCUSDT` (snapshot + delta, nominal 20 ms);
- `publicTrade.BTCUSDT` (real-time batches, exploded to trades);
- `tickers.BTCUSDT` (nominal 100 ms sparse snapshot/delta; OI, funding, next
  funding time and independent L1 quote).

Application ping is sent every 20 seconds. The raw client is used because the
high-level `pybit` order-book callback reconstructs the book and rewrites message
semantics. Bybit documents `u`, `seq`, and `cts`, but not a contiguous `+1`
invariant or a previous-update ID. Therefore jumps are diagnostics, not proof of
loss. Proven transport/local loss, parse failure, queue overflow, decreasing or
conflicting identities, an invalid/crossed book, or reconnect invalidate the
book until a fresh snapshot. Optional public REST metadata/backfill endpoints
are `/v5/market/instruments-info`, `/v5/market/open-interest`, and
`/v5/market/funding/history`; none blocks the pilot.

## 3. Immutable raw schemas

Every raw part also has `schema_version`, `session_id`, `connection_id`,
`local_message_no`, `recv_utc_ns`, `recv_monotonic_ns`, `payload_hash`,
`raw_json`, and `parse_ok`. Original price/quantity strings are retained in the
JSON; derived tables use `float64`.

### `data/raw/orderbook/<session>/part-*.parquet` (one wire message per row)

`topic`, `type`, `exchange_ts_ms`, `matching_ts_ms`, `symbol`, `update_id`,
`cross_seq`, `bids_json`, `asks_json`, `n_bid_updates`, `n_ask_updates`,
`state_action`, `book_valid_after`, `duplicate`, `sequence_anomaly`, `error`.

### `data/raw/trades/<session>/part-*.parquet` (one trade per row)

`message_id`, `trade_index`, `exchange_ts_ms`, `trade_ts_ms`, `symbol`,
`taker_side`, `price`, `quantity`, `tick_direction`, `trade_id`,
`is_block_trade`, `is_rpi_trade`, `cross_seq`, `duplicate`, `error`.

### `data/raw/aux/<session>/part-*.parquet` (one sparse ticker/event row)

`record_kind`, `event_type`, `topic`, `type`, `exchange_ts_ms`, `cross_seq`,
`symbol`, `present_fields_json`, nullable `bid1_price`, `bid1_size`,
`ask1_price`, `ask1_size`, `last_price`, `mark_price`, `index_price`,
`open_interest`, `open_interest_value`, `funding_rate`,
`next_funding_time_ms`, `funding_interval_hour`, `reconnect_index`, `detail`,
`error`.

Parts are written to a same-directory temporary file and atomically renamed to a
new unique name. A session manifest records start/end, config, topics, host clock
and collector counters. Existing raw files are never replaced.

## 4. Processed research-table schema

`data/processed/<session>/research_table.parquet` has one row for every exact UTC
grid second. Availability is determined by `recv_utc_ns <= timestamp`; exchange
time is diagnostic only. Invalid rows remain in the grid.

Identity/quality:

- `timestamp`, `session_id`, `symbol`, `book_valid`, `usable`,
  `book_age_ms`, `book_exchange_ts_ms`, `book_recv_utc_ns`, `update_id`,
  `cross_seq`, `book_depth_bid`, `book_depth_ask`, `gap_or_resync_interval`.

Book state (for `i=01..20`):

- `bid_price_i`, `bid_size_i`, `ask_price_i`, `ask_size_i`.

Price and book features:

- `best_bid`, `best_ask`, `mid`, `spread_abs`, `spread_bps`, `microprice`,
  `microdev`;
- for `k in {1,5,10,20}`: `bid_depth_k`, `ask_depth_k`, `obi_k`;
- `ofi_event_sum_1s` and trailing `ofi_{1,5,10,30}s`, with full-window
  coverage flags.

Trade-flow features for `w in {1,5,10,30,60}`:

- `buy_volume_ws`, `sell_volume_ws`, `signed_volume_ws`,
  `volume_imbalance_ws`, `buy_count_ws`, `sell_count_ws`,
  `trade_count_imbalance_ws`, `total_trade_volume_ws`, `trade_count_ws`, and
  `trade_coverage_ws`.

Price controls and auxiliary state:

- `return_{1,5,10,30,60}s`, `realized_vol_{10,30,60}s`;
- nullable causal ticker state plus per-field age: `ticker_bid1/ask1`,
  `open_interest`, `open_interest_delta_60s`, `funding_rate`,
  `next_funding_time_ms`, `time_to_funding_s`, and `ticker_age_ms`.

For each `H in {10,30,60}`:

- `target_end_time_Hs`, `future_mid_Hs`, `future_bid_Hs`, `future_ask_Hs`,
  `target_return_Hs`, `target_class_Hs` (`-1/0/+1`),
  `target_valid_Hs`, `target_path_valid_Hs`.

Targets are exact timestamp self-joins (`t+H`), never nearest matches. The
directional dead zone is independent of all trading cost scenarios.

## 5. Leakage controls

1. Keep the complete grid and choose all boundaries before filtering rows.
2. Features use local availability time and `(t-W,t]` trailing windows only;
   no future interpolation or backfill.
3. Exact `t+H` targets require valid endpoints; invalid paths are separately
   flagged and excluded from primary trading diagnostics.
4. Freeze a common maximum-horizon purge and embargo at every boundary. Source
   labels must end strictly before the next segment; the destination begins only
   after the configured embargo.
5. Walk-forward folds are expanding and preserve order. Imputer, scaler, model,
   feature orientation, hyperparameters, regime/quantile cutpoints and trading
   threshold are fit/selected on that fold's past only.
6. The final 20% is not used by EDA selection, ablation selection, thresholds,
   model choice, cost choice or verdict thresholds. It is evaluated once after
   those choices are frozen.
7. Model feature lists are explicit and reject every `target_`, `future_` or
   split/output column. Append-future prefix-invariance tests protect the feature
   builder.
8. Every run writes boundaries, embargo, features, seed, package versions,
   dataset SHA-256 and configuration. `LEAKAGE_CHECKLIST.md` documents the same
   contract for manual audit.

## 6. Analysis and outputs

- Data quality is generated before modeling as JSON, one-row CSV and Markdown.
  Structural failures stop modeling; low coverage/sample size remains visible
  and forbids a scientific GO.
- Models: zero, expanding historical mean and previous-return controls; OLS,
  Ridge (tiny fixed alpha grid), multinomial logistic regression, and one
  LightGBM model (tiny fixed grid). No neural model or broad search.
- Pairwise-complete descriptive/IC/MI tables are produced overall, by fold, day
  and hour. IID p-values are labelled approximate; confirmatory correlation
  significance uses HAC with lag at least `H-1`. BH-FDR families are declared
  separately for pre-test discovery and final confirmation.
- Quantile edges, feature orientation, volatility/spread regime cutpoints,
  extreme-event cutoff and non-overlapping trading threshold are frozen from
  development/validation only. Ties may collapse bins.
- Taker PnL uses `log(bid_exit/ask_entry)` for long and
  `log(bid_entry/ask_exit)` for short. The visible spread is already included;
  configured cost scenarios are explicitly incremental round-trip fee/slippage
  bps. Mid and taker diagnostics are labelled separately.
- Moving-block bootstrap uses paired calendar-time blocks of at least the target
  horizon. HAC/Newey-West estimates are reported without absurd short-sample
  annualization.
- Group comparisons are fixed as price-history, order-book, trade-flow,
  order-book+trade-flow and all features. Extreme 1% removal and first/second
  half checks are descriptive robustness tables.
- All requested raw CSV/JSON/Parquet tables, at least the 20 named PNG plots,
  predictions/trades, a regenerated `report.md`, config/environment/fingerprint
  manifests and explicit omissions are stored under a unique run directory.

## 7. Predeclared verdict

The primary confirmatory pair is set in YAML (`ridge`, 30 s by default); “best
model/horizon” is descriptive and cannot replace it. A run shorter than the
configured research minimum (12 h by default), with too few final-test rows or
critical data-quality failure receives `INSUFFICIENT DATA — SMOKE ONLY`, never a
GO.

For an adequately sized run, evidence flags and their numeric thresholds all
live in YAML: positive primary untouched-test IC/rank-IC, same sign in most
walk-forward folds, monotone OOS quantiles, non-concentrated contribution,
positive microstructure-vs-price-only lift, a positive time-series-aware
uncertainty result, and survival under the predeclared non-zero incremental cost.
`GO` requires the core statistical/stability/microstructure evidence plus broad
support, including economic evidence. `WEAK GO` requires stable statistical and
microstructure evidence but may fail costs or uncertainty. Otherwise the result
is `NO-GO`. Every flag and underlying number is printed so the user can override
the heuristic judgment.

## 8. Resource budget

For immutable ZSTD Parquet containing depth-50 wire payloads, trades and sparse
ticker messages, expect roughly 0.08-0.30 GB/hour in ordinary conditions and up
to about 0.6 GB/hour during bursts. Typical raw totals are therefore 0.08-0.30
GB (1 h), 1.0-3.6 GB (12 h), and 1.9-7.2 GB (24 h). Reserve approximately 1, 10
and 20 GB respectively for raw parts, processed data, run artifacts and safety
headroom.

On a 4-8 core laptop with streamed raw parts: collection should remain below
0.5 GB RAM; analysis is expected to peak near 1-1.5 GB (1 h), 1.5-3 GB (12 h),
and 2-4 GB (24 h). Approximate CPU runtimes are 2-8, 10-35 and 20-70 minutes.
Bootstrap count, LightGBM threads and raw part size remain configurable for a
lower-memory machine.

## 9. Build and verification order

1. Configuration, immutable IO and book state machine.
2. Raw collector and data-quality audit.
3. Causal grid/features/targets and purged splits.
4. Statistics/models/evaluation/trading diagnostics.
5. Plots/report/run manifest and one-command CLI.
6. Unit and synthetic end-to-end tests.
7. Short live mainnet smoke capture, raw/reconstruction audit, dataset build,
   smoke-only full pipeline, artifact inspection and report regeneration.

