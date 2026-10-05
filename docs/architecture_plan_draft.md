# Architecture plan: `microstructure_alpha_probe`

Status: implementation plan, frozen before looking at pilot results. Reference repository inspected read-only at commit [`933ddbb57b5a987e3f0313e0483611d04d915bbc`](https://github.com/Mifikcha/Transformer_RV_Crypto/tree/933ddbb57b5a987e3f0313e0483611d04d915bbc). Current Bybit protocol details are sourced in [`reference_research.md`](reference_research.md).

## 1. Small-scope design

The pilot is one venue, one instrument, one recorder, and one regular 1-second research table. It is a signal-existence test, not a production bot. There is no database, Docker, neural network, live execution, multi-exchange framework, or broad hyperparameter search.

Command surface:

```text
python scripts/run_pilot.py collect --hours 12
python scripts/run_pilot.py collect                 # until Ctrl+C
python scripts/run_pilot.py analyze [--session SESSION_ID]
python scripts/run_pilot.py report --run RUN_ID
```

Proposed modules and ownership:

| Module | One responsibility |
|---|---|
| `src/config.py` | Load YAML, apply defaults, validate all horizons/costs/thresholds, emit a fully resolved config. |
| `src/collect.py` | Raw WebSocket frames, receive timestamps, bounded queue, heartbeat/reconnect, immutable Parquet parts, session event journal. |
| `src/book.py` | Explicit `UNSYNCED/SYNCED` depth-50 reconstruction and validation. |
| `src/build_dataset.py` | Stream raw parts, reconstruct state, create the complete 1-second UTC receive-time grid, causal joins, quality masks. |
| `src/features.py` | Only the predeclared price, depth, trade-flow, OFI, return, volatility, and optional auxiliary features. |
| `src/targets.py` | Exact-timestamp future mid targets, class labels, and bid/ask exit support columns. |
| `src/splits.py` | Outer 60/20/20 chronological boundaries, inner expanding folds, purge/embargo assertions. |
| `src/models.py` | Naive, OLS, Ridge, logistic regression, and one LightGBM model with tiny fixed grids. |
| `src/evaluate.py` | Fold/test regression and classification metrics, quantiles, group ablations, importance. |
| `src/statistics.py` | Pairwise statistics, HAC estimates, BH-FDR, moving-block bootstrap, regime/concentration checks. |
| `src/trading_diagnostic.py` | Frozen-threshold, single-position, fixed-horizon optimistic/taker diagnostics. |
| `src/diagnostics.py` | Raw/session data-quality summaries and required plots. |
| `src/report.py` | Deterministic `report.md` and verdict from already-written tables only. |
| `scripts/run_pilot.py` | Thin CLI orchestration; no research logic. |

`analyze` executes: freeze config and raw-part manifest -> hash input -> reconstruct and quality-report -> build causal table/targets -> freeze time boundaries -> fit/evaluate development candidates -> write `decision_lock.json` -> load final test once -> diagnostics/tables/plots -> verdict/report. A fatal integrity failure still creates a run folder and quality report, but does not train models.

## 2. What to reuse from `Transformer_RV_Crypto`

Adapt these patterns: UTC-normalized timestamps and deterministic chronological sort/dedup; stable numeric dtypes; chunked Parquet; symbol-aware centralized paths; sorted backward `merge_asof`; trailing (not centered) rolling windows with denominator checks; explicit sanity checks; scaler fitted independently on each training fold; expanding walk-forward evaluation with a maximum-horizon purge; semantic feature groups, held-out permutation importance, ablations, PNG output, and both human/machine-readable logs. Primary-source file citations are in [`reference_research.md`](reference_research.md).

Do **not** copy: deleting an existing output before write, unrestricted forward fill, any backward fill, hard-coded 8-hour funding clock, hard-coded fee/horizon, row shifts without exact timestamp validation, zero-filling before EDA, warning-only gaps, or feature selection over the eventual test tail. In particular, the old repository's `bfill()` of trailing HAR features is future leakage; early warm-up rows here remain missing/invalid.

Everything microstructure-specific is new: lossless WebSocket recording, reconnect/proven-loss state machine, depth reconstruction, receive-time sampling, book/trade features, quality interval masks, executable quote accounting, time-series uncertainty/concentration diagnostics, final-test lock, and predeclared verdict.

## 3. Current Bybit inputs

Use the unauthenticated mainnet linear endpoint `wss://stream.bybit.com/v5/public/linear` and one subscription request for:

- `orderbook.50.BTCUSDT`: currently the smallest supported depth covering L20; nominal 20 ms push. Snapshot replaces the book. Delta size `0` deletes, a new price inserts, and an existing price replaces size. Persist `type`, `ts`, `cts`, `data.s/b/a/u/seq` unchanged. [Official order-book protocol](https://bybit-exchange.github.io/docs/v5/websocket/public/orderbook)
- `publicTrade.BTCUSDT`: real-time batches, potentially 1,024 trades/message. Persist `T,s,S,v,p,L,i,BT,RPI,seq`; `S` is taker side. Multiple messages may share `seq`, so trade identity is `(symbol, i)`, not `seq`. [Official trade protocol](https://bybit-exchange.github.io/docs/v5/websocket/public/trade)
- `tickers.BTCUSDT` (optional but default-on capture, non-blocking for analysis): nominal 100 ms snapshot/sparse-delta updates with best quotes, OI, funding, and next funding time. Omitted delta fields mean unchanged. [Official ticker protocol](https://bybit-exchange.github.io/docs/v5/websocket/public/ticker)

Public streams require no key. Send application `{"op":"ping"}` every 20 s, require successful subscription acknowledgement, and reconnect with capped exponential backoff plus jitter. [Official connection guidance](https://bybit-exchange.github.io/docs/v5/ws/connect)

Do not use the high-level `pybit` order-book callback for raw capture: it reconstructs internally and rewrites callbacks as synthetic snapshots. Use raw frames through `websocket-client`. The JSON V5 order-book page documents `u` and cross-topic `seq`, but neither a previous-update pointer nor a `+1` invariant. Therefore:

- never report `u != prior_u + 1` as a proven gap;
- enter `UNSYNCED` on disconnect, local queue drop, decode/write failure, contradictory replay, or decreasing/out-of-order evidence;
- store deltas while `UNSYNCED` but do not apply/publish them;
- reconnect/resubscribe and wait for a valid fresh snapshot before publishing state;
- report ID jumps separately. `sequence_gap_count` means **provable continuity breaks** under this definition, and the report records the definition; unsupported numeric inference is never shown as zero gaps.

Auxiliary REST endpoints are optional cross-check/backfill only: `/v5/market/open-interest`, `/v5/market/funding/history`, and `/v5/market/instruments-info`. Funding timing comes from `nextFundingTime`/instrument `fundingInterval`, never a hard-coded clock.

## 4. Immutable raw layout and exact schemas

Raw parts live at `data/raw/{orderbook,trades,aux}/date=YYYY-MM-DD/session_id=<id>/part-000000.parquet`. Rotate every 5 minutes or 128 MiB. Create `*.tmp`, fsync/close, atomically rename to a never-before-used part name, and fail on collision. Raw rows are never deduplicated or rewritten. Each part gets SHA-256, row count, byte count, min/max receive time in `session_manifest.json`. `raw_text` is the authoritative exact UTF-8 frame; typed fields make inspection efficient. Malformed frames remain as rows with null typed fields and `parse_ok=false`.

Common fields in all three datasets:

| Column | Arrow type | Null? | Meaning |
|---|---:|:---:|---|
| `schema_version` | `uint16` | no | Raw schema version, initially `1`. |
| `session_id` | `string` | no | One CLI recording invocation. |
| `connection_id` | `string` | no | Changes after every reconnect. |
| `local_message_no` | `uint64` | no | Strict receive order within session. |
| `recv_utc_ns` | `int64` | no | UTC wall clock captured at callback entry. |
| `recv_monotonic_ns` | `int64` | no | Local spacing/staleness clock. |
| `topic` | `string` | yes | Wire topic. |
| `message_type` | `string` | yes | Wire `type` or control type. |
| `exchange_ts_ms` | `int64` | yes | Wire envelope `ts`. |
| `raw_text` | `large_string` | no | Exact received text frame. |
| `payload_sha256` | `fixed_size_binary[32]` | no | SHA-256 of exact frame bytes. |
| `parse_ok` | `bool` | no | Typed-field parse status. |
| `parse_error` | `string` | yes | Bounded error code/message. |

Order-book additions, one row per wire message:

| Column | Arrow type |
|---|---:|
| `matching_ts_ms` (`cts`) | `int64?` |
| `symbol` (`data.s`) | `string?` |
| `depth` | `int16?` |
| `update_id` (`data.u`) | `int64?` |
| `cross_seq` (`data.seq`) | `int64?` |
| `bids` (`data.b`) | `list<struct<price:string,size:string>>?` |
| `asks` (`data.a`) | `list<struct<price:string,size:string>>?` |

Trade additions, one row per wire message (the builder explodes the nested list, preserving indices):

| Column | Arrow type |
|---|---:|
| `trade_count` | `uint16?` |
| `trades` (`data[]`) | `list<struct<trade_ts_ms:int64,symbol:string,taker_side:string,quantity:string,price:string,tick_direction:string,trade_id:string,is_block_trade:bool,is_rpi_trade:bool,cross_seq:int64>>?` |

Ticker/aux additions, one row per sparse message:

| Column | Arrow type |
|---|---:|
| `cross_seq` (`cs`) | `int64?` |
| `symbol` | `string?` |
| `present_fields` | `list<string>?` |
| `bid1_price`, `bid1_size`, `ask1_price`, `ask1_size`, `open_interest`, `open_interest_value`, `funding_rate`, `next_funding_time_ms`, `funding_interval_hour`, `mark_price`, `index_price` | `string?` except `next_funding_time_ms:int64?` |

`data/raw/aux/.../session-events-part-*.parquet` has `schema_version,session_id,connection_id,event_no,utc_ns,monotonic_ns,event_type,topic,severity,details_json`; event types include connect, subscription ack/fail, ping/pong, disconnect, reconnect, queue drop, parse error, write failure, `UNSYNCED`, snapshot accepted, and resynchronized. This journal is the source of reconnect/gap/resync counts.

Raw prices/quantities remain decimal strings. Derived tables use `float64`. A duplicate is marked only in derived processing: exact book/ticker replay uses connection/topic/IDs/hash; trades deduplicate on `(symbol, trade_id)`. Conflicting duplicate IDs invalidate continuity. Block/RPI-trade inclusion is config-driven and both flags are always retained.

## 5. Book state, availability clock, and data quality

The research clock is **local receive time**, not exchange event time. Grid row `t` contains only messages whose `recv_utc_ns <= t`; this avoids granting the model information before it reached the recorder. Exchange `ts/cts/T` remain ordering/latency diagnostics. Record wall and monotonic clocks, flag wall-clock regressions, and label `recv_utc - exchange_ts` as apparent latency (host clock synchronization is not assumed).

Book state machine:

1. Start/reconnect `UNSYNCED`.
2. A snapshot replaces both sides. Accept it only if prices/sizes parse, sizes are positive, sides are sorted, at least 20 levels exist each side, and `best_bid < best_ask`; then enter `SYNCED`.
3. In `SYNCED`, apply each delta by exact decimal-string price key, then validate positive sizes and uncrossed best quotes. Any validation/local-loss failure enters `UNSYNCED` immediately.
4. Any later snapshot, including `u=1`, replaces state and increments resynchronization count.
5. A second is book-usable only while `SYNCED` and `book_age_ms <= quality.max_book_age_ms` (default 1,000 ms, explicitly a pilot policy, reported in config). Flow features are invalid for their entire lookback after any proven capture gap; zero observed trades on a healthy feed means zero flow, whereas a disconnected trade feed means missing.

`data_quality.json` is the rich nested source, `data_quality.csv` is one row per session/stream, and `data_quality_report.md` explains failures. They contain at minimum: first/last exchange and receive timestamps, duration; total messages/trades/book updates; snapshot/delta/reconnect/proven-gap/resync/exact-duplicate/malformed counts; ID-jump and out-of-order counts; queue drops/write failures; missing interval count/total/longest; median/p90/p95/p99 monotonic event spacing by stream; apparent `recv-ts` and `recv-cts/T` latency min/p1/p5/median/p90/p95/p99/max plus negative count; stale-book periods/count/duration; crossed/invalid/insufficient-depth counts; ticker-vs-book best-quote disagreement; usable seconds and percentage by split. Bad quality is printed before model metrics and displayed prominently in the report.

Fatal modeling conditions: no accepted snapshot, corrupt/unhashable raw part, unresolved queue/write loss without subsequent snapshot, no nonempty purged folds, or no exact valid target rows. Other bad intervals are excluded by explicit masks, never repaired.

## 6. Exact processed research schema

Write `data/processed/<dataset_id>/research_table.parquet`, unique and immutable, one row for **every** UTC second in the requested interval. Do not drop invalid seconds from this canonical table. Arrow types are `timestamp[ns, tz=UTC]` for `ts`, `float64` for numeric market/features/targets, `int64` for IDs/counts/ages (nullable where noted), `bool` for masks, and `int8` for class labels (`-1=short, 0=flat, +1=long`).

Provenance/quality columns:

```text
ts, session_id, connection_id,
book_recv_ts_ns, book_exchange_ts_ms, book_matching_ts_ms,
book_update_id, book_cross_seq, book_age_ms,
book_valid, depth20_valid, trade_feed_valid, ticker_valid,
gap_in_lookback_60s, feature_valid,
ticker_recv_ts_ns, ticker_exchange_ts_ms, ticker_age_ms
```

Current book state and depth features:

```text
best_bid, best_ask, mid, spread_abs, spread_bps, microprice, microdev,
bid_depth_l1, ask_depth_l1, obi_l1,
bid_depth_l5, ask_depth_l5, obi_l5,
bid_depth_l10, ask_depth_l10, obi_l10,
bid_depth_l20, ask_depth_l20, obi_l20
```

Definitions are fixed:

```text
mid = (best_bid + best_ask) / 2
spread_bps = 1e4 * (best_ask - best_bid) / mid
microprice = (best_ask * bid_size_l1 + best_bid * ask_size_l1)
             / (bid_size_l1 + ask_size_l1)
microdev = (microprice - mid) / spread_abs
obi_k = (bid_depth_k - ask_depth_k) / (bid_depth_k + ask_depth_k)
```

Zero/nonpositive denominators or spreads produce null plus invalid flags; never epsilon-distort a value into validity.

For every `w in {1,5,10,30,60}` seconds, exact trailing interval `(t-w,t]`, create:

```text
buy_volume_{w}s, sell_volume_{w}s, signed_volume_{w}s,
volume_imbalance_{w}s,
buy_trade_count_{w}s, sell_trade_count_{w}s,
trade_count_imbalance_{w}s, total_trade_volume_{w}s
```

Buy-taker quantity is positive and sell-taker quantity negative. An imbalance denominator of zero yields `0` only when the feed is healthy and there truly were no trades; otherwise null.

Top-of-book OFI uses the documented Cont-style event contribution, evaluated only across continuous valid book states:

```text
e_n = I(b_n >= b_prev) q^b_n - I(b_n <= b_prev) q^b_prev
    - I(a_n <= a_prev) q^a_n + I(a_n >= a_prev) q^a_prev
ofi_{w}s = sum(e_n for recv time in (t-w,t]), w in {1,5,10,30}
```

Recent price controls:

```text
return_1s, return_5s, return_10s, return_30s, return_60s,
realized_vol_10s, realized_vol_30s, realized_vol_60s,
ofi_1s, ofi_5s, ofi_10s, ofi_30s
```

`return_ws = log(mid_t/mid_{t-w})` requires exact valid endpoints and no capture gap. Realized volatility is the square root of the sum of squared exact 1-second log returns over a fully valid trailing window.

Optional auxiliary columns (capture absence cannot block the core table/model):

```text
funding_rate, open_interest, open_interest_delta_60s,
next_funding_time_ms, time_to_funding_s, aux_feature_valid
```

Ticker sparse fields are causally carried forward only after first observation, with source time/age retained and a configured maximum age. `time_to_funding_s` comes from `nextFundingTime`; no backward fill or assumed 00/08/16 UTC schedule. Optional auxiliary features enter models only if the rule frozen from the training split says their coverage is sufficient; otherwise the whole aux group is omitted and the reason recorded.

For every `H in {10,30,60}` (and `300` only when enabled), add support/target columns:

```text
future_mid_{H}s, future_bid_{H}s, future_ask_{H}s,
target_return_{H}s, target_class_{H}s, target_valid_{H}s
```

The future row must have timestamp exactly `t+H`; a next-available quote is forbidden. `target_return_Hs = log(future_mid_Hs/mid)`. `target_class` uses configured `dead_zone_bps`: `+1` above, `-1` below negative threshold, else `0`. `target_valid` requires feature validity at `t`, valid current/future quotes, and no known capture-invalid interval through `t+H`. All `future_*` and `target_*` columns are in an explicit denylist and are asserted absent from model matrices.

## 7. Leakage controls and reproducible splitting

`LEAKAGE_CHECKLIST.md` documents and tests all of the following:

- Build the complete grid and freeze time boundaries before filtering by target/model outcomes. No shuffle or random split.
- Outer intervals are chronological `[0%,60%)` train, `[60%,80%)` validation, `[80%,100%]` final test by elapsed time, not by surviving rows.
- At every boundary, accept a row in the earlier split only when its exact target end is strictly before the next split start. `embargo_rows = ceil(max_horizon_seconds / sample_period_seconds)` is additionally asserted; with 1 s and Hmax 60 it is 60 rows (300 only if enabled).
- Within the outer 60% training interval, split time into six contiguous blocks and run five expanding folds: block 0 trains fold 0, block 1 validates; blocks 0-1 train fold 1, block 2 validates; etc. Purge each training tail by the same exact target-end rule.
- The validation interval alone chooses the finite candidate/model/horizon and trading threshold. The final model is trained on the purged train 60%, not silently refit after threshold calibration. This deliberately sacrifices some sample efficiency for an unambiguous pilot.
- A `sklearn.Pipeline` fits median imputation (when needed) and scaling on each training slice only. Linear/logistic models scale; LightGBM need not. Missing targets are never imputed. EDA uses pairwise deletion and reports `n`.
- Trailing windows are right-closed and past-only; no centered rolling, future interpolation, or `bfill`. Backward/as-of joins require `source_recv_time <= t` and a maximum age.
- Regime cut points, quantile/signal bucket edges, near-zero-variance decisions, optional-aux inclusion, hyperparameters, feature groups, and trade threshold are learned/frozen without test data.
- Before reading test targets/predictions, write `decision_lock.json` containing selected horizon/model/features/hyperparameters/preprocessor, regimes, threshold, cost reference, verdict rules, split timestamps, seed, and its SHA-256. Test evaluation is one command path and cannot rerun selection.
- Test feature/target tables may be reported after the lock, but cannot change any choice. `report` only re-renders saved tables.

Static feature groups are `price_history`, `order_book`, `trade_flow`, `order_book+trade_flow`, and `all`; auxiliary is a separately disclosed optional addition. No significance-driven feature selection is needed for this small pilot.

## 8. Models and statistical diagnostics

Required deterministic candidates per horizon:

- zero, train historical mean, and previous-return-sign/return naive baselines;
- OLS and Ridge (`alpha` in a tiny frozen grid such as `{0.1,1,10}`);
- multinomial logistic regression (`C` in `{0.1,1,10}`) for directional targets;
- one LightGBM regressor/classifier family, at most three predeclared shallow configurations, fixed seed and CPU thread count.

The validation champion for the verdict is selected only among all-feature OLS/Ridge/LightGBM regression candidates and 10/30/60 s horizons. Deterministic sort key is highest validation rank IC, then lowest RMSE, then fixed simplicity order OLS -> Ridge -> LightGBM and horizon order 30 -> 60 -> 10. Candidates must first have positive median inner-fold rank IC and at least 4/5 same-direction folds. Classification, other horizons, and ablations remain diagnostics and cannot replace a failed champion after test inspection.

For each feature, write the requested count/missing/mean/std/min/percentiles/max/skew/excess-kurtosis/near-zero-variance and configured lag autocorrelations. For each feature-horizon and split/fold/day/hour where sample size permits, write Pearson, Spearman, mutual information, sign, effective `n`, and HAC-aware approximate p-values for linear/rank association. MI has no invented analytic p-value. Apply Benjamini-Hochberg separately to a declared family `(split, association_type)` across every feature x horizon; no feature is called a discovery on raw `p<0.05` alone.

Key IC and strategy mean intervals use a paired moving-block bootstrap with 2,000 repetitions and fixed seed. Default block is `5 * max_horizon` seconds (300 s for Hmax 60), because overlapping H-second targets induce dependence; sensitivity blocks of 1x and 15x Hmax are reported. Bootstrap code accumulates summaries rather than materializing a large `n_boot x n` array. HAC/Newey-West uses the chronological series and a predeclared lag at least H. Never use IID bootstrap.

Quantiles are deciles from frozen development edges. Report counts, mean/median/std return, hit rate, long/short accuracy, top-minus-bottom spread and bootstrap CI, and Spearman/Kendall monotonicity of bucket rank versus bucket mean. All fold metrics remain individual rows plus mean/std. Regression/classification/trading metrics are exactly those requested in the task; multiclass ROC-AUC/PR-AUC are macro one-vs-rest with per-class values, and Brier is multiclass mean squared probability error with the definition saved.

Volatility regimes use train-only 33rd/67th percentiles of `realized_vol_60s`; optional spread regimes use train-only tertiles. Concentration includes hour/day/regime contributions, best/worst periods, top 1/5/10% trade contribution, best-hour/day share, and leave-best-hour/top-1%-trades sensitivity. Extreme-event robustness drops the top 1% absolute target returns using a cutoff learned on development and then applied unchanged to test. First/second test halves are always shown when nonempty.

## 9. Trading diagnostic definition

Avoid overlapping pseudo-trades: hold at most one unit-notional position. When flat, enter on a frozen threshold; ignore signals until the exact fixed-H exit. Validation selects one threshold from the predeclared absolute-prediction quantiles `{0, 0.50, 0.70, 0.80, 0.90}`, with minimum validation trade count, under the reference cost. Freeze the corresponding numeric threshold before test.

For signal `s in {-1,+1}`:

```text
optimistic long gross = log(mid[t+H] / mid[t])
optimistic short gross = log(mid[t] / mid[t+H])
taker long gross = log(bid[t+H] / ask[t])
taker short gross = log(bid[t] / ask[t+H])
net = gross - extra_round_trip_cost_bps / 10_000
```

Thus taker gross already includes quoted spread crossing. Cost scenarios `[0,2,5,10,15]` bps mean additional round-trip fee/slippage assumptions, not claims about Bybit fees. `reference_cost_bps=5` is a configurable conservative feasibility policy. Report gross and net separately for both executions, and never call mid-price results executable. PnL is the sum of log returns; turnover, exposure, holding time, drawdown, win rate, profit factor, nonannualized mean/std Sharpe-like statistic, Sortino, and counts are explicit. Annualization is suppressed for datasets under 30 calendar days and otherwise still caveated.

## 10. Predeclared verdict

All numeric policies below live in YAML and are echoed in `decision_lock.json`; they are feasibility thresholds, not universal market truths. Always expose the underlying numbers and sensitivity. Smoke runs do not claim alpha.

Eligibility gates for any positive verdict:

1. At least 12 recorded hours, at least 2 usable final-test hours, five nonempty inner folds, >=90% usable seconds in each outer interval, no unresolved integrity failure, and >=100 completed test trades for economic claims.
2. The champion and all choices were locked before test access.

If eligibility fails, required label is `NO-GO` with reason code `INSUFFICIENT_EVIDENCE` or `DATA_INTEGRITY`; the report explicitly says this is not evidence that alpha is absent.

Core statistical gates for the locked champion:

1. Test Spearman IC is at least `0.01` (documented pilot effect-size floor) and its 95% moving-block-bootstrap lower bound is above zero.
2. Outer validation IC is positive, inner folds have the same positive sign in at least 4/5 folds, and median fold IC is positive.
3. Test decile mean versus decile rank Spearman is >=0.80, top-decile mean exceeds bottom-decile mean, and the block-bootstrap CI for the top-bottom difference excludes zero on the positive side.
4. IC remains positive in both test halves, after the development-frozen top-1%-extreme cutoff, and after removal of the best contributing test hour.
5. The same locked model/horizon with all microstructure features has higher test IC than its price-history-only ablation. At minimum the point difference must be positive; a positive lower bound on the paired block-bootstrap difference is the stronger standard.

Economic gates use the non-overlapping taker diagnostic at frozen threshold and 5 bps additional round-trip cost: positive net PnL and mean trade, 95% block-bootstrap lower bound above zero, >=100 trades, and positive net after separately removing the best hour and top 1% of trades.

- **GO**: eligibility + all core gates + strong (CI-positive) microstructure increment + all economic gates.
- **WEAK GO**: eligibility + core gates 1-4 + positive point microstructure increment, but its uncertainty includes zero and/or the economic gate fails. This means reproducible statistical structure worth a focused follow-up, not a profitable strategy.
- **NO-GO**: eligibility failure, core statistical failure, no positive incremental microstructure value, unstable signs/quantiles, or apparent performance dependent on the extreme/best-hour episode. Train success or gross mid-price PnL can never promote the verdict.

The 0.01/0.80/90%/sample policies are intentionally visible and configurable; sensitivity tables include IC floors 0.005/0.02, quintiles versus deciles, bootstrap blocks, and every cost scenario. The automated label is a screen, not a substitute for reading the evidence tables.

## 11. Reproducibility and run outputs

Unique folder `runs/YYYY-MM-DD_HHMMSS_btcusdt_<short-hash>/`, created exclusively. Required contents:

```text
config.yaml                       # fully resolved, not merely user overrides
manifest.json                     # run id, UTC times, CLI, OS/CPU/Python, git commit/dirty flag
environment.txt                  # sorted installed package versions
dataset_manifest.json            # raw part paths, hashes, sizes, rows, overall fingerprint
decision_lock.json               # choices and test-access lock
feature_list.json
split_boundaries.json
LEAKAGE_CHECKLIST.md
analysis.log
metrics_summary.csv / .json
fold_metrics.csv
feature_statistics.csv
feature_target_correlations.csv
feature_stability.csv
multiple_testing.csv
quantile_analysis.csv
classification_metrics.csv
trading_metrics.csv
cost_sensitivity.csv
regime_metrics.csv
ablation_metrics.csv
bootstrap_confidence_intervals.csv
predictions_test.parquet
trades_test.parquet
data_quality.json / .csv
data_quality_report.md
verdict.json
report.md
plots/*.png
```

Randomness uses one master seed expanded deterministically per fold/model/bootstrap. Record LightGBM threads because parallel reductions can vary slightly. Dataset fingerprint is SHA-256 of a canonical manifest containing every immutable raw part's content SHA-256, relative path, rows, size, and schema version; part hashes are computed once at close, so analyzing a 24-hour dataset does not need to reread all bytes merely to name it. Record data interval and exact feature/split lists. `report --run` reads saved artifacts only and must reproduce identical Markdown without retraining.

## 12. Resource budget (Windows laptop)

Assumptions: BTC depth-50 up to 50 book pushes/s, variable batched trades, ticker up to 10 pushes/s, exact frame plus typed columns, ZSTD Parquet, five-minute/128 MiB parts. Activity and JSON compressibility vary greatly, so collection records actual bytes/hour and warns from observed extrapolation.

| Recording | Typical raw | Burst/stress raw | Suggested free space before start | Analysis wall time (4-8 CPU cores) | Peak analysis RAM |
|---:|---:|---:|---:|---:|---:|
| 1 h | 0.08-0.30 GB | up to 0.6 GB | 1 GB | 2-8 min | 1-1.5 GB |
| 12 h | 1.0-3.6 GB | up to 7.2 GB | 10 GB | 10-35 min | 1.5-3 GB |
| 24 h | 1.9-7.2 GB | up to 14.4 GB | 20 GB | 20-70 min | 2-4 GB |

Run tables/PNGs and the 1-second processed table add roughly 0.05-0.8 GB depending on duration and plot/prediction output. Collection itself should stay below roughly 0.5 GB because the queue and Parquet batches are bounded. Analysis must stream raw parts and aggregate to seconds; loading all wire frames into pandas is forbidden. The upper runtime grows to roughly 1-2 hours on a slow laptop if all 2,000-replicate robustness bootstraps and repeated permutation importance are enabled. GPU is neither used nor required.

These are capacity estimates, not guarantees. The collector checks free disk before start, every part rotation, and stops cleanly before a configurable reserve rather than corrupting a part.

## 13. Minimal Windows dependencies

Target CPython 3.11 x64 and pinned wheels in `requirements.txt`:

- `numpy`, `pandas`, `pyarrow` (Arrow/Parquet with ZSTD);
- `PyYAML`;
- `websocket-client` (raw V5 frames; no `pybit` mutation);
- `scipy`, `scikit-learn`, `statsmodels` (models, MI, metrics, HAC, BH-FDR);
- `lightgbm` (single tree family; current PyPI provides Windows wheels);
- `matplotlib` (Agg backend; seaborn is unnecessary);
- `pytest` (development/test dependency may be split into `requirements-dev.txt`).

Use the standard library for CLI, JSON, hashes, logging, paths, timestamps, threading/queues, and the few optional public REST GETs. Do not add ccxt, PyTorch, SHAP, a database, or XGBoost. If the pinned LightGBM wheel cannot install, fail with an explicit environment error; do not silently substitute another model and make runs incomparable.

## 14. Minimum tests

Use tiny deterministic fixtures and one synthetic end-to-end table. Required checks: snapshot replacement; delta insert/update/delete; later snapshot and `u=1` reset; `UNSYNCED` after proven loss and no feature publication until snapshot; unsupported `u+1` assumption absent; duplicate trade ID versus shared trade `seq`; sparse ticker field carry-forward; exact receive-time cutoff; no future interpolation/backfill; flow invalidation through lookback after gaps; exact `t+H` target alignment; class thresholds; outer chronological boundaries; purge/embargo at every fold; train-only imputer/scaler; feature denylist; long/short optimistic and taker PnL signs; spread and extra-cost accounting; immutable part collision; manifest hash reproducibility. The synthetic toy set must have hand-computable book, OFI, trade imbalance, targets, and PnL, and must fail if a future row enters a feature.

## 15. Scientific limitation to state in every short run

A 1-hour collection is only a plumbing smoke test. Twelve to twenty-four hours can produce a **preliminary** feasibility read but cannot cover days, weekends, macro regimes, or exchange changes; even a GO is only permission to collect a substantially longer preregistered sample. RPI orders are absent from the ordinary public order book, receive-latency depends on the host/network/clock, quote-based taker prices are not guaranteed fills, and overlapping return labels reduce effective sample size. These limitations belong at the top of `report.md`, not in fine print.
