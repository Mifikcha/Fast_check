# Reference research: `Transformer_RV_Crypto` and Bybit V5 public microstructure feeds

Research date: 2026-09-10. The reference repository was inspected read-only at commit [`933ddbb57b5a987e3f0313e0483611d04d915bbc`](https://github.com/Mifikcha/Transformer_RV_Crypto/tree/933ddbb57b5a987e3f0313e0483611d04d915bbc). Bybit protocol statements below come from current official V5 documentation and the official SDK source; links point to the owning primary sources.

## Bottom line

- For BTCUSDT linear perpetual, connect without credentials to `wss://stream.bybit.com/v5/public/linear` and subscribe to `orderbook.50.BTCUSDT`, `publicTrade.BTCUSDT`, and `tickers.BTCUSDT`. Public topics require no authentication. Bybit currently offers depth 1, 50, 200, and 1000 for linear/inverse; depth 50 is the smallest supported depth that satisfies the requested top-20-to-50 pilot and is pushed nominally every 20 ms. [Bybit Connect](https://bybit-exchange.github.io/docs/v5/ws/connect) and [Bybit Orderbook](https://bybit-exchange.github.io/docs/v5/websocket/public/orderbook)
- Persist the actual wire messages before reconstruction. Do **not** use the high-level `pybit.WebSocket.orderbook_stream` callback as the raw recorder: the official SDK applies deltas internally, changes every order-book callback's `type` to `snapshot`, and passes the reconstructed book. This destroys raw snapshot/delta provenance. [Official `pybit` implementation, pinned commit](https://github.com/bybit-exchange/pybit/blob/e6a557985396757830ee0ed031d3254abe398d0a/pybit/_websocket_stream.py#L446-L563)
- Bybit V5 currently documents `u` (update ID), `seq` (cross sequence), and `cts` (matching-engine timestamp), but it does **not** document that either ID is gap-free or must increase by exactly one, and the V5 payload has no documented previous-update-ID field. Therefore `u != last_u + 1` is not a defensible gap rule. A collector can prove loss from a disconnect, local queue overflow/drop, parse/write failure, or clear out-of-order/replay evidence; after any such event it should mark the book unusable, reconnect/resubscribe, and wait for a fresh `snapshot`. This is a conservative inference from the fields and rules the official specification actually provides. [Bybit Orderbook response fields](https://bybit-exchange.github.io/docs/v5/websocket/public/orderbook#response-parameters)
- Use the ticker stream for convenient live best bid/ask, open interest, funding rate, and next funding time. Use REST OI/funding endpoints for backfill or cross-checking, not as a dependency of the core 10-60 second pilot. Ticker derivatives updates are nominally 100 ms and are sparse deltas: an omitted field means “unchanged,” not missing. [Bybit Ticker](https://bybit-exchange.github.io/docs/v5/websocket/public/ticker)
- Reuse the old repository's UTC normalization, chronological sort/dedup, backward as-of joins, train-only preprocessing, expanding-window evaluation, horizon embargo, feature-family ablations, held-out permutation importance, Parquet chunking, and explicit output/log tables. Do not reuse its output overwrite behavior, backward fills, hard-coded funding clock/fees/horizon, global zero-imputation for EDA, warning-only treatment of gaps, or absence of a separately untouched final test.

## What is reusable from `Transformer_RV_Crypto`

### Reuse or adapt

1. **UTC and deterministic ordering.** `get_OHLCV.py` parses naive timestamps as UTC, converts aware timestamps to UTC, casts numeric fields to stable `float64`, sorts chronologically, and deduplicates timestamps before persistence. Those are sound ingestion-boundary patterns. [`get_OHLCV.py`](https://github.com/Mifikcha/Transformer_RV_Crypto/blob/933ddbb57b5a987e3f0313e0483611d04d915bbc/dataset/get_data/get_OHLCV.py#L48-L65) and [normalization/dedup](https://github.com/Mifikcha/Transformer_RV_Crypto/blob/933ddbb57b5a987e3f0313e0483611d04d915bbc/dataset/get_data/get_OHLCV.py#L128-L140)
2. **Chunked Parquet writing.** The fetcher writes time-bounded chunks and stitches them with `pyarrow.parquet.ParquetWriter`, which keeps memory bounded. For the new collector, use immutable session/part files instead of stitching over a raw destination. [`get_OHLCV.py`](https://github.com/Mifikcha/Transformer_RV_Crypto/blob/933ddbb57b5a987e3f0313e0483611d04d915bbc/dataset/get_data/get_OHLCV.py#L146-L193)
3. **Centralized, symbol-aware paths.** `_paths.py` keeps symbol normalization and stage paths in one place. The small probe should keep that idea in YAML plus a few path helpers, without the old pipeline's many intermediate stages. [`_paths.py`](https://github.com/Mifikcha/Transformer_RV_Crypto/blob/933ddbb57b5a987e3f0313e0483611d04d915bbc/dataset/get_data/_paths.py)
4. **Backward-only alignment.** `merge_dataset.py` and `add_funding.py` use sorted `merge_asof(..., direction="backward")`; this is the right causal primitive. The new code must additionally impose a per-source tolerance/maximum age and retain source timestamp/age columns, so stale auxiliary data is visible. [`merge_dataset.py`](https://github.com/Mifikcha/Transformer_RV_Crypto/blob/933ddbb57b5a987e3f0313e0483611d04d915bbc/dataset/get_data/merge_dataset.py#L49-L64) and [`add_funding.py`](https://github.com/Mifikcha/Transformer_RV_Crypto/blob/933ddbb57b5a987e3f0313e0483611d04d915bbc/dataset/get_data/add_funding.py#L63-L77)
5. **Causal rolling expressions.** `add_volume_statistics.py` uses ordinary trailing pandas windows, not centered windows, and protects z-scores against zero denominators. Keep the pattern, but use time-based windows or a rigorously regular 1-second table and explicit `min_periods`. [`add_volume_statistics.py`](https://github.com/Mifikcha/Transformer_RV_Crypto/blob/933ddbb57b5a987e3f0313e0483611d04d915bbc/dataset/get_data/add_volume_statistics.py#L22-L52)
6. **Structural sanity checks.** `validate_OHLCV.py` checks required columns, UTC timestamps, sorting, duplicate timestamps, interval regularity, and price consistency, failing on invalid OHLC. Translate that style to bid/ask positivity, sorted sides, uncrossed book, supported message type/topic, sequence diagnostics, stale ages, and exact target timestamps. [`validate_OHLCV.py`](https://github.com/Mifikcha/Transformer_RV_Crypto/blob/933ddbb57b5a987e3f0313e0483611d04d915bbc/dataset/get_data/validate_OHLCV.py#L48-L98)
7. **Train-only preprocessing and expanding folds.** Baselines fit `StandardScaler` on each fold's training slice and only transform validation. `walk_forward_split` uses expanding training data and removes the maximum forward-label horizon from the end of every training fold. Preserve both concepts, while defining the embargo from configured seconds/rows rather than parsing target names. [`baselines/utils.py`](https://github.com/Mifikcha/Transformer_RV_Crypto/blob/933ddbb57b5a987e3f0313e0483611d04d915bbc/baselines/utils.py#L254-L303), [`group_ablation.py`](https://github.com/Mifikcha/Transformer_RV_Crypto/blob/933ddbb57b5a987e3f0313e0483611d04d915bbc/feature_selection/group_ablation.py#L79-L96)
8. **Interpretable evidence artifacts.** The old EDA writes tabular summaries and non-interactive PNGs; semantic feature groups feed both permutation importance and group ablation; fold outputs remain inspectable instead of being reduced to one score. This is a strong pattern for the probe. [`eda.py`](https://github.com/Mifikcha/Transformer_RV_Crypto/blob/933ddbb57b5a987e3f0313e0483611d04d915bbc/feature_selection/eda.py#L76-L203), [`feature_groups.py`](https://github.com/Mifikcha/Transformer_RV_Crypto/blob/933ddbb57b5a987e3f0313e0483611d04d915bbc/feature_selection/feature_groups.py), [`permutation_importance.py`](https://github.com/Mifikcha/Transformer_RV_Crypto/blob/933ddbb57b5a987e3f0313e0483611d04d915bbc/feature_selection/permutation_importance.py#L72-L117), and [`group_ablation.py`](https://github.com/Mifikcha/Transformer_RV_Crypto/blob/933ddbb57b5a987e3f0313e0483611d04d915bbc/feature_selection/group_ablation.py#L108-L156)
9. **Human and machine-readable logging.** The training entry point emits config and per-fold metrics as JSON-shaped log records while writing timestamped file logs; the feature-selection pipeline writes dedicated CSVs plus a plain-text log. Reuse the dual human/machine-readable idea inside a unique run directory. [`run_transformer.py`](https://github.com/Mifikcha/Transformer_RV_Crypto/blob/933ddbb57b5a987e3f0313e0483611d04d915bbc/transformer/run_transformer.py#L39-L65) and [`run_feature_selection.py`](https://github.com/Mifikcha/Transformer_RV_Crypto/blob/933ddbb57b5a987e3f0313e0483611d04d915bbc/feature_selection/run_feature_selection.py#L54-L142)

### Do not copy these patterns

1. **Silent destruction of an existing output.** `fetch_full_range` removes the requested output and its temporary chunk directory if they already exist. That contradicts immutable raw capture. Create a unique session directory and atomically create new numbered part files; refuse collisions. [`get_OHLCV.py`](https://github.com/Mifikcha/Transformer_RV_Crypto/blob/933ddbb57b5a987e3f0313e0483611d04d915bbc/dataset/get_data/get_OHLCV.py#L209-L225)
2. **Backward fill.** `transformer/dataset.py` calls `bfill()` on trailing HAR features, so early rows receive values computed later. That is explicit future leakage. Do not backward-fill any model input; leave warm-up rows missing, add missingness/age indicators where justified, and fit any imputer on training data only. [`transformer/dataset.py`](https://github.com/Mifikcha/Transformer_RV_Crypto/blob/933ddbb57b5a987e3f0313e0483611d04d915bbc/transformer/dataset.py#L35-L58)
3. **Hard-coded funding schedule.** `add_funding.py` assumes settlement at 00:00/08:00/16:00 UTC. Bybit explicitly says funding interval differs by symbol; query `fundingInterval`/`fundingIntervalHour` and use `nextFundingTime`. [`add_funding.py`](https://github.com/Mifikcha/Transformer_RV_Crypto/blob/933ddbb57b5a987e3f0313e0483611d04d915bbc/dataset/get_data/add_funding.py#L79-L93), [Bybit funding history](https://bybit-exchange.github.io/docs/v5/market/history-fund-rate), and [Bybit instruments info](https://bybit-exchange.github.io/docs/v5/market/instrument)
4. **Hard-coded target/cost assumptions.** `add_target.py` fixes a 60-minute horizon, 5-minute row step, and `C=0.0005`. The new project must obtain horizons and all cost/dead-zone scenarios from the frozen run config. [`add_target.py`](https://github.com/Mifikcha/Transformer_RV_Crypto/blob/933ddbb57b5a987e3f0313e0483611d04d915bbc/dataset/get_data/add_target.py#L14-L36)
5. **Row-shift without timestamp validation.** The old target is `close.shift(-H_STEPS)`. That is safe only after proving a complete regular grid. In the probe, require an exact `t+H` timestamp and usable books at both endpoints; never allow a missing interval to shorten the economic horizon. [Old target implementation](https://github.com/Mifikcha/Transformer_RV_Crypto/blob/933ddbb57b5a987e3f0313e0483611d04d915bbc/dataset/get_data/add_target.py#L21-L38)
6. **Zero-imputing before EDA.** `eda.py` fills both features and targets with zero before correlations and mutual information. This changes the empirical distribution and can manufacture dependence around missing periods. Pairwise-drop missing observations for univariate statistics; report `n`, missingness, and the imputation policy separately. [`eda.py`](https://github.com/Mifikcha/Transformer_RV_Crypto/blob/933ddbb57b5a987e3f0313e0483611d04d915bbc/feature_selection/eda.py#L88-L126)
7. **Warning-only gaps and unlimited forward fill.** The validator merely reports irregular steps, and the merge pipeline forward-fills without a tolerance. For microstructure data, known gaps/staleness must invalidate rows and flow into the quality report. [`validate_OHLCV.py`](https://github.com/Mifikcha/Transformer_RV_Crypto/blob/933ddbb57b5a987e3f0313e0483611d04d915bbc/dataset/get_data/validate_OHLCV.py#L78-L94), [`merge_dataset.py`](https://github.com/Mifikcha/Transformer_RV_Crypto/blob/933ddbb57b5a987e3f0313e0483611d04d915bbc/dataset/get_data/merge_dataset.py#L59-L72)
8. **No separately untouched final test in feature selection.** The shared walk-forward function eventually uses the tail of the full dataframe as a validation fold, and EDA/importance/ablation call it over the full dataset. The feasibility probe needs an outer chronological 60/20/20 split first, feature/model/threshold decisions confined to the first 80%, and a one-shot final 20% evaluation. [`baselines/utils.py`](https://github.com/Mifikcha/Transformer_RV_Crypto/blob/933ddbb57b5a987e3f0313e0483611d04d915bbc/baselines/utils.py#L254-L303) and [`eda.py`](https://github.com/Mifikcha/Transformer_RV_Crypto/blob/933ddbb57b5a987e3f0313e0483611d04d915bbc/feature_selection/eda.py#L120-L139)

## Current Bybit V5 public protocol

### Connection, subscription, and liveness

- Mainnet linear public endpoint: `wss://stream.bybit.com/v5/public/linear`; testnet: `wss://stream-testnet.bybit.com/v5/public/linear`. Public topics do not require authentication. [Official connection table](https://bybit-exchange.github.io/docs/v5/ws/connect)
- Subscribe with `{"op":"subscribe","args":["orderbook.50.BTCUSDT","publicTrade.BTCUSDT","tickers.BTCUSDT"],"req_id":"<local-id>"}` and require a successful command response (`success: true`, `op: "subscribe"`) before considering the subscription active. [Official subscription examples and response](https://bybit-exchange.github.io/docs/v5/ws/connect#how-to-subscribe-to-topics)
- Send the application heartbeat `{"op":"ping"}` every 20 seconds, observe the `pong`, and reconnect promptly on disconnect. Bybit warns that network disconnects can occur at any time. [Official heartbeat and reconnect guidance](https://bybit-exchange.github.io/docs/v5/ws/connect#how-to-send-the-heartbeat-packet)
- Give every transport connection a new local `connection_id`/`session_id`; count reconnects explicitly. Immediately set book state to `UNSYNCED` on transport loss. Resubscribe and do not publish reconstructed observations until a new order-book snapshot arrives.

### Order book: `orderbook.50.BTCUSDT`

For linear/inverse, Bybit documents depth/frequency pairs `1/10 ms`, `50/20 ms`, `200/100 ms`, and `1000/200 ms`. Level 1 is snapshot-only; depth 50 supplies both the snapshot and deltas needed for the pilot. RPI orders are not included, so this feed is not literally every executable order on the venue. [Official depth table and RPI caveat](https://bybit-exchange.github.io/docs/v5/websocket/public/orderbook#depths)

Wire envelope and data fields:

| Field | Meaning / storage consequence |
|---|---|
| `topic` | Expected exact topic, e.g. `orderbook.50.BTCUSDT`. |
| `type` | `snapshot` or `delta`; persist unchanged from the wire. |
| `ts` | Bybit system generation timestamp in milliseconds. |
| `data.s` | Symbol. |
| `data.b`, `data.a` | Arrays of `[price_string, size_string]`; bids in a snapshot are descending, asks ascending. In deltas they are changed levels, not a full side. |
| `data.u` | Update ID. `u=1` can accompany a service-restart snapshot and requires a full overwrite. |
| `data.seq` | Cross sequence; a smaller value means the data was generated earlier and it can compare generation order across depth feeds. |
| `cts` | Matching-engine production timestamp in ms; Bybit says it can be correlated with trade field `T`. |

Source: [official order-book schema and examples](https://bybit-exchange.github.io/docs/v5/websocket/public/orderbook#response-parameters).

Book application rules are exact: a new `snapshot` replaces the whole local book; for a delta, size `0` deletes the price, a new price inserts it, and an existing price replaces its size. If another snapshot arrives later, reset again. [Official snapshot/delta rules](https://bybit-exchange.github.io/docs/v5/websocket/public/orderbook#process-snapshotdelta)

Recommended conservative state machine:

1. Start/reconnect as `UNSYNCED`; record all wire messages, but do not expose book features.
2. On a valid `snapshot`, replace both sides, validate sorted positive sizes, `best_bid < best_ask`, record `u/seq/cts`, and enter `SYNCED`.
3. On `delta` while `SYNCED`, apply the exact price-level operations, then validate the reconstructed top of book. On `delta` while `UNSYNCED`, store it but do not apply/publish it.
4. On any later snapshot, including service restart `u=1`, replace the book and increment the resynchronization counter.
5. On a provable local loss (bounded-queue overflow, decoding/write failure), disconnect/reconnect and remain invalid until the next snapshot. The documentation does not support pretending that an arbitrary `u` jump proves a missing message.
6. Treat decreasing `seq` as out-of-order generation evidence; retain and count exact-payload replays. Do not demand `seq == previous + 1`: `seq` is cross-topic generation order, not documented as a per-topic contiguous counter.

Store both a wall-clock UTC receive timestamp and `time.monotonic_ns()` at the callback boundary. `receive_utc_ms - cts` and `receive_utc_ms - ts` are useful latency proxies only if the host clock is synchronized; report negative values/clock jumps rather than clipping them. Monotonic time is the correct basis for local spacing and stale timers.

### Public trades: `publicTrade.BTCUSDT`

The channel is real-time. Each envelope has `topic`, `type` (documented as `snapshot` even though these are incremental trade batches), system timestamp `ts`, and `data[]` sorted by trade time ascending. Each trade contains:

- `T`: match timestamp ms;
- `s`: symbol;
- `S`: taker side `Buy` or `Sell` (the aggressor direction needed for signed flow);
- `v`: size string;
- `p`: price string;
- `L`: tick direction for perps/futures;
- `i`: trade ID;
- `BT`: block-trade flag;
- `RPI`: RPI-trade flag;
- `seq`: cross sequence.

Bybit explicitly warns that a futures/spot message may contain up to 1,024 trades and multiple messages can have the same `seq`. Therefore explode all `data[]` records, preserve `message_index`/`trade_index`, and deduplicate by trade ID `i` (plus symbol), **not by `seq`**. [Official public-trade protocol](https://bybit-exchange.github.io/docs/v5/websocket/public/trade)

Decide in config whether block/RPI trades participate in features; never silently discard them. Retain the flags in raw data so both definitions can be analyzed.

### Ticker/auxiliary stream: `tickers.BTCUSDT`

Derivatives tickers are nominally pushed every 100 ms and use `snapshot` plus sparse `delta`. If a response parameter is absent, its value has not changed. Persist raw sparse messages, and separately maintain a causal state cache with per-field `last_update_ts`/age. [Official ticker semantics](https://bybit-exchange.github.io/docs/v5/websocket/public/ticker)

Relevant envelope/data fields are `topic`, `type`, `cs` (cross sequence), `ts`, `data.symbol`, `bid1Price`, `bid1Size`, `ask1Price`, `ask1Size`, `openInterest`, `openInterestValue`, `fundingRate`, `nextFundingTime`, `fundingIntervalHour`, `markPrice`, and `indexPrice`. [Official linear ticker fields](https://bybit-exchange.github.io/docs/v5/websocket/public/ticker#response-parameters)

The ticker's best quote is useful as an independent quality cross-check, but the reconstructed depth-50 book remains the source of research features. A disagreement should be measured and flagged, not repaired by looking forward.

### REST open interest, funding, and instrument metadata

| Purpose | Public request | Important fields/rules |
|---|---|---|
| OI history | `GET https://api.bybit.com/v5/market/open-interest?category=linear&symbol=BTCUSDT&intervalTime=5min` | `intervalTime` is required and starts at 5 minutes (`5min`, `15min`, `30min`, `1h`, `4h`, `1d`); optional `startTime`, `endTime`, `limit` 1-200 and `cursor`; response `list[].openInterest`, `singleOpenInterest`, `timestamp`, `nextPageCursor`. For BTCUSDT linear, OI size unit is BTC. Bybit warns this endpoint may lag during extreme volatility. [Official OI endpoint](https://bybit-exchange.github.io/docs/v5/market/open-interest) |
| Settled funding history | `GET https://api.bybit.com/v5/market/funding/history?category=linear&symbol=BTCUSDT` | Optional `startTime`, `endTime`, `limit` 1-200; passing `startTime` alone is an error. Response has `list[].fundingRate` and `fundingRateTimestamp`. Intervals differ by symbol. [Official funding-history endpoint](https://bybit-exchange.github.io/docs/v5/market/history-fund-rate) |
| Contract metadata | `GET https://api.bybit.com/v5/market/instruments-info?category=linear&symbol=BTCUSDT` | Response includes `status`, `contractType`, `priceFilter.tickSize`, `lotSizeFilter.qtyStep`, and `fundingInterval` in minutes. [Official instrument endpoint](https://bybit-exchange.github.io/docs/v5/market/instrument) |

The live official instrument response observed on 2026-09-10 identified BTCUSDT as `LinearPerpetual`, `Trading`, with `fundingInterval: 480` minutes, but this should be recorded as run metadata rather than hard-coded. [Live official BTCUSDT instrument response](https://api.bybit.com/v5/market/instruments-info?category=linear&symbol=BTCUSDT)

## Suggested lossless raw columns

These columns follow directly from the documented wire fields while adding local provenance. Preserve the complete raw JSON (or an equivalent Arrow struct) so parser migrations are possible.

### `data/raw/orderbook/<session>/part-*.parquet` — one row per wire message

`session_id`, `connection_id`, `local_message_no`, `recv_utc_ns`, `recv_monotonic_ns`, `topic`, `type`, `exchange_ts_ms` (`ts`), `matching_ts_ms` (`cts`), `symbol`, `update_id` (`u`), `cross_seq` (`seq`), `bids` (list of price/size pairs), `asks` (list of price/size pairs), `raw_json`, `payload_hash`, `parse_ok`.

### `data/raw/trades/<session>/part-*.parquet` — one row per trade

`session_id`, `connection_id`, `local_message_no`, `trade_index`, `recv_utc_ns`, `recv_monotonic_ns`, `exchange_ts_ms` (`ts`), `trade_ts_ms` (`T`), `symbol`, `taker_side` (`S`), `price` (`p`), `quantity` (`v`), `tick_direction` (`L`), `trade_id` (`i`), `is_block_trade` (`BT`), `is_rpi_trade` (`RPI`), `cross_seq` (`seq`), `raw_json` or a message-payload reference.

### `data/raw/aux/<session>/part-*.parquet` — one row per sparse ticker message

`session_id`, `connection_id`, `local_message_no`, `recv_utc_ns`, `recv_monotonic_ns`, `topic`, `type`, `exchange_ts_ms` (`ts`), `cross_seq` (`cs`), `symbol`, nullable documented ticker fields, `present_fields`, `raw_json`, `payload_hash`, `parse_ok`. Keeping `present_fields` distinguishes “unchanged/omitted” from an explicit empty value.

Use decimal or original string representation for prices and quantities in raw data; convert to `float64` only in derived research tables. Partition by session and bounded part size/time, create files exclusively, and write through a temporary file followed by an atomic rename. Never merge or overwrite raw parts during collection.

## Protocol caveats that should become tests or quality flags

- A newly received snapshot is normal resynchronization behavior and must reset the book; it is not a delta to merge.
- `u=1` indicates a service-restart snapshot and must reset local state.
- `orderbook.1` repeats snapshot with the same `u` after three seconds without change and is snapshot-only; use depth 50 for this project.
- RPI orders are excluded from the ordinary order-book stream, so “public order book” has a documented coverage limitation.
- Trade `seq` is not a unique message or trade identifier; multiple messages may share it.
- Ticker deltas omit unchanged fields; naive row-wise null interpretation will create artificial missingness.
- REST OI has no sub-five-minute history, making it secondary for 10-60 second prediction; live ticker OI can be retained, with age/staleness diagnostics.
- Funding intervals are symbol-specific. Derive the schedule from instrument/ticker metadata, and distinguish the live ticker rate from a settled historical funding record.
- The official high-level Python SDK mutates order-book and ticker callbacks into reconstructed snapshots. Raw scientific capture needs a lower-level WebSocket client or raw frame hook.
- Bybit gives ordering metadata but no documented contiguous-update invariant. Report only gaps that can actually be established; do not manufacture a clean “zero sequence gaps” statistic from an unsupported `+1` assumption.

