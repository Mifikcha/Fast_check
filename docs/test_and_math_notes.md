# Verification and mathematical contract

This document is a pre-implementation contract for the dangerous parts of the
microstructure feasibility probe. It is intentionally stricter than a normal
backtest. A test that fails because data cannot be proved valid is preferable to
silently manufacturing a usable row.

## 1. Non-negotiable conventions

1. All timestamps are timezone-aware UTC. Store exchange event time, local wall
   receive time, and a process-monotonic receive/ingest sequence separately.
2. **Availability is determined by local receive time**, not exchange event
   time. At grid time `t`, a feature may only use messages with
   `local_receive_time <= t`. Exchange timestamps remain useful for event-time
   windows and latency diagnostics, but cannot move a late-arriving message into
   the past.
3. All intervals below use half-open notation unless explicitly stated. A
   rolling window of width `W` at `t` is `(t-W, t]`.
4. Raw numeric strings should be parsed to finite positive prices and finite
   non-negative quantities. Invalid values invalidate the affected update; do
   not coerce them to zero.
5. A missing value means "unknown/unusable". Zero means a known zero. In
   particular, invalid book/trade coverage must not be encoded as neutral OFI or
   neutral imbalance.
6. Grid rows are never removed before target construction or time splitting.
   Keep invalid rows with flags and `NaN` values, then filter by explicit masks.
7. Fix one random seed per run and write it, all temporal boundaries, all
   feature names, and all statistical-family definitions into the run output.

The Bybit V5 public linear endpoint currently documents the topics
`orderbook.50.BTCUSDT` and `publicTrade.BTCUSDT`. The official order-book page
states that snapshots replace the entire local book; delta quantity zero deletes
a price, a new price inserts it, and an existing price updates it. It describes
`u` as the update ID and `seq` as a cross-sequence useful for temporal comparison.
It does **not** state as clearly for the public JSON feed that every adjacent
`seq` is consecutive. Therefore `seq + 1` must not be a gap rule. References:

- <https://bybit-exchange.github.io/docs/v5/websocket/public/orderbook>
- <https://bybit-exchange.github.io/docs/v5/websocket/public/trade>
- <https://bybit-exchange.github.io/docs/v5/ws/connect>

## 2. Order-book state machine

### 2.1 States

Use three explicit states per connection/topic:

- `WAITING_FOR_SNAPSHOT`: no trustworthy book; ignore deltas for derived state.
- `VALID`: a snapshot was accepted and subsequent deltas are consistent.
- `INVALID`: a gap, malformed update, crossed/empty book, local queue loss,
  disconnect, or staleness was detected. Ignore deltas until a new snapshot.

Raw messages are persisted in every state. Reconnect immediately transitions to
`WAITING_FOR_SNAPSHOT`; state from the old TCP connection must not leak into the
new connection.

### 2.2 Snapshot and delta rules

For a valid `snapshot`:

1. Parse the whole payload into fresh bid and ask maps keyed by price.
2. Reject duplicate price levels within one side unless the protocol has a
   documented last-update rule (it currently does not).
3. Validate positive prices, non-negative sizes, non-empty sides, and
   `best_bid < best_ask`.
4. Atomically replace both sides and record `u`, `seq`, timestamps, connection
   ID, and ingest sequence. Any snapshot replaces the old book even if its IDs
   reset. Bybit documents `u == 1` as a service-restart snapshot.
5. Enter `VALID` only after all checks pass.

For a delta while `VALID`, apply all changes atomically to a copy, then validate
the resulting book before commit:

- `size == 0`: delete the price if present; deleting an absent price is an
  idempotent no-op but should be counted for diagnostics.
- `size > 0` and price absent: insert.
- `size > 0` and price present: replace size, never add to the old size.
- Sort bids descending and asks ascending only when materializing levels; map
  storage itself need not be sorted.
- Empty or crossed output (`best_bid >= best_ask`) is invalid, not a row with a
  negative spread.

Do not calculate an OFI increment across a snapshot/resynchronization boundary.

### 2.3 IDs, duplicates, gaps, and reordering

Use `u` as the topic update ID and `seq` only as a monotonic cross-stream
diagnostic.

- Exact repeated `(connection_id, topic, type, u, seq, payload_hash)` is a
  duplicate: count and ignore it without changing state.
- Same update identity with a different payload is corruption: invalidate.
- A delta with an older ID is out-of-order. TCP should preserve order, so the
  conservative action is to invalidate, not retroactively apply it.
- A new snapshot always resets state, including an unsolicited one.
- `seq` may be required to increase, but a jump larger than one is **not by
  itself** proof of a missing topic update; cross-sequences can include other
  feeds.
- For the selected JSON depth-50 feed, treating `u > last_u + 1` as a hard gap is
  scientifically conservative, but its consecutiveness must be confirmed from
  the current protocol or a clean smoke capture. If the guarantee remains
  ambiguous, force resynchronization on a jump and record the policy/version.
  False invalidation loses data; accepting a real gap manufactures book state.
  The former is the safer feasibility-probe error.
- `u == last_u` can only be ignored when the full payload is identical. The
  documented repeated level-1 snapshot behavior must not be generalized to
  depth 50 without evidence.

On a hard gap, disconnect, or detected local queue overflow, mark an invalid
interval beginning at detection time and resubscribe/reconnect for a fresh WebSocket
snapshot. A REST snapshot cannot safely be spliced into a live delta stream
unless deltas are buffered and update IDs are explicitly bridged; do not add
that complexity to the pilot.

### 2.4 Staleness and validity at the 1-second grid

At each grid tick, materialize the last `VALID` state whose receive time is at or
before the tick. Also store `book_age_ms = grid_time - last_book_receive_time`.
If age exceeds configured `max_book_age_ms`, or the state machine was invalid at
the tick, set `book_valid = false` and all book-derived features to `NaN`.

Staleness is not the same as an unchanged book. For BTC depth 50, long silence is
suspicious, but the threshold must be configurable and reported. If heartbeat,
trade, or ticker traffic is alive while the book topic is silent, log this as a
topic-specific stale period.

### 2.5 Minimal book tests

1. **Snapshot reconstruction**: bids `[(100,2),(99,3)]`, asks
   `[(101,4),(102,5)]` yields best bid 100, best ask 101, depth two each, and a
   valid book.
2. **Replacement, insertion, deletion**: delta bids `[(100,5),(99,0),(98,7)]`
   leaves `{100:5,98:7}`. Quantity at 100 must be 5, not 7.
3. **Atomic multi-side delta**: update both sides in one message and assert no
   intermediate state is emitted.
4. **Gap**: snapshot `u=10`, then delta `u=12` enters `INVALID`; the delta is not
   reflected. Another delta is ignored. A subsequent snapshot `u=20` fully
   restores `VALID`.
5. **Duplicate**: applying the same `u=11` delta twice produces the same book and
   increments duplicate count once.
6. **Conflicting duplicate**: same `u` with different payload invalidates.
7. **Unsolicited snapshot/reset**: a fresh snapshot discards levels present only
   in the prior state. `u=1` is accepted as a reset.
8. **Crossed or empty result**: deletion of the last ask, or best bid equal to
   best ask, invalidates and emits no usable state.
9. **Reconnect**: after a disconnect, a delta cannot revive the old book; only a
   snapshot can.

## 3. Feature definitions and causal construction

Let best bid/ask prices be `b_t`, `a_t`, with quantities `q^b_t`, `q^a_t`.

```text
mid_t        = (b_t + a_t) / 2
spread_t     = a_t - b_t
spread_bps_t = 10,000 * spread_t / mid_t
microprice_t = (a_t * q^b_t + b_t * q^a_t) / (q^b_t + q^a_t)
microdev_t   = (microprice_t - mid_t) / spread_t
```

The microprice uses contra-side weighting: more bid quantity moves it toward the
ask. If either price is invalid, total L1 quantity is zero, or spread is at/below
the configured numeric epsilon, return `NaN`, not zero. With positive sizes,
`microdev` should lie in `[-0.5, 0.5]` up to floating error.

For top `k` levels:

```text
B_k   = sum(base_quantity of first k bids)
A_k   = sum(base_quantity of first k asks)
OBI_k = (B_k - A_k) / (B_k + A_k)
```

If either side has fewer than `k` valid levels or the denominator is zero,
`OBI_k` is `NaN`; do not silently turn `OBI_20` into variable-depth imbalance.
Store actual depth and `B_k`, `A_k`. Expected finite range is `[-1,1]`.

For trades received in `(t-W,t]`, Bybit `S=Buy` means buyer/taker initiated and
gets sign `+1`; `S=Sell` gets `-1`:

```text
buy_volume       = sum(q_i where side_i = Buy)
sell_volume      = sum(q_i where side_i = Sell)
signed_volume    = buy_volume - sell_volume
volume_imbalance = signed_volume / (buy_volume + sell_volume)
count_imbalance  = (n_buy - n_sell) / (n_buy + n_sell)
```

When the collector is healthy but there are no trades in the window, set the two
imbalances and totals to zero and retain a `has_trades/count` indicator. When the
underlying coverage is invalid, use `NaN`. Deduplicate by trade ID before
aggregation; one WebSocket envelope may contain many trades.

Lagged return at lag `L` uses exact grid endpoints:

```text
lag_return_L(t) = log(mid_t / mid_(t-L))
```

Realized volatility over `W` seconds is unannualized:

```text
RV_W(t) = sqrt(sum_{j=t-W+step}^{t} one_step_log_return_j^2)
```

It requires `W / step` valid one-step returns (and therefore one more valid mid
endpoint), unless a separately configured coverage rule is explicitly reported.

### 3.1 Prefix-invariance leakage test

For every causal feature builder, build features from an event prefix ending at
`T`, then append arbitrary events strictly after `T` and rebuild. Every feature
row at or before `T` must be byte/equality-consistent (allowing only documented
floating tolerance). This one property catches centered windows, backward fill,
future interpolation, and many accidental forward joins.

Toy data: a stable book/trade sequence through `t=2`, then an enormous price and
volume jump at `t=3`. Features at `t=0,1,2` must be identical whether the `t=3`
event exists or not.

## 4. Canonical top-of-book OFI

Use the standard event-level top-of-book OFI, calculated once per accepted
atomic book delta. Let previous/current top quotes be `(P^b_0,Q^b_0,P^a_0,Q^a_0)`
and `(P^b_1,Q^b_1,P^a_1,Q^a_1)`.

Bid contribution:

```text
e_bid = +Q^b_1             if P^b_1 > P^b_0
        Q^b_1 - Q^b_0      if P^b_1 = P^b_0
        -Q^b_0             if P^b_1 < P^b_0
```

Ask-side supply contribution:

```text
e_ask = +Q^a_1             if P^a_1 < P^a_0
        Q^a_1 - Q^a_0      if P^a_1 = P^a_0
        -Q^a_0             if P^a_1 > P^a_0
```

Then `OFI_event = e_bid - e_ask`. Thus a stronger/improved bid is positive, an
improved/larger ask is negative, and removal/worsening of the ask is positive.
Aggregate by summing accepted event increments received in `(t-W,t]`.

Important rules:

- Compute from event states, not only consecutive 1-second snapshots; otherwise
  changes that reverse within a second vanish.
- A delta touching only deeper levels can correctly have zero L1 OFI.
- Do not bridge a snapshot, reconnect, invalid interval, or stale interval.
- If any part of the OFI window lacks trustworthy book coverage, primary OFI is
  `NaN`. Optionally report a separate coverage fraction, but never call partial
  coverage a full-window value.
- Raw OFI is in base-asset quantity units. If a normalized variant is added,
  give it a separate name and denominator; do not silently rescale this feature.

Minimal OFI cases, starting from bid `100@2`, ask `101@3`:

| Current top | Expected OFI |
|---|---:|
| bid `100@5`, ask `101@1` | `+5` (`+3 - (-2)`) |
| bid improves to `100.5@4`, ask unchanged | `+4` |
| bid worsens to `99.5@7`, ask unchanged | `-2` |
| bid unchanged, ask improves to `100.5@4` | `-4` |
| bid unchanged, ask worsens to `101.5@6` | `+3` |

Also assert that the sum of event OFI values inside `(t-W,t]` obeys exact
right-closed boundary behavior: an event exactly at `t-W` is excluded and one at
`t` is included.

## 5. Target alignment

For horizon `H`:

```text
target_end_time = t + H
r_(t,H) = log(mid_(t+H) / mid_t)
```

Both endpoints must be valid, non-stale, and strictly positive. Join on the
**exact regular-grid timestamp** `t+H`. Never use nearest or future as-of
matching for the target. Never apply `shift(-H_rows)` after dropping invalid or
missing rows: that silently changes a 10-second target into the next ten retained
observations. `shift` is acceptable only after a complete grid is reindexed and
invalid rows remain present, but an explicit timestamp self-join is safer.

Store at least `target_end_time`, endpoint-valid flags, and a
`horizon_contains_invalid_interval` diagnostic. The primary endpoint return can
be defined when both endpoints are valid even if an intermediate book interval
was invalid, but trading/path diagnostics should expose and preferably exclude
such cases. Whichever primary policy is chosen must be fixed before seeing model
results.

Directional classes use an independently configured dead zone `delta_H`:

```text
short (-1) if r_(t,H) < -delta_H
flat   (0) if -delta_H <= r_(t,H) <= delta_H
long  (+1) if r_(t,H) > +delta_H
```

Values exactly on either boundary are flat. If configured in bps, record the
conversion. A clear convention is `delta_H = dead_zone_bps / 10,000` in log-return
units (the small-return approximation); alternatively use `log1p(bps/10,000)`,
but do not mix the two. Classification dead zones and trading transaction costs
are different configuration fields.

Minimal alignment toy:

- Complete 1-second grid mids `[100,101,102,103,104]` at times `0..4` and `H=2`
  gives targets `log(102/100)`, `log(103/101)`, `log(104/102)`, `NaN`, `NaN`.
- Mark time 2 invalid. Targets at time 0 and time 2 are invalid because an
  endpoint is invalid; time 1 remains `log(103/101)` under the endpoint-only
  policy but is flagged as containing an invalid interval.
- Remove physical row 2 before construction and assert that time 0 does **not**
  get paired with time 3 or 4.
- With `delta=0.001`, returns `-0.001`, `0`, and `+0.001` are all flat; only
  values strictly outside are short/long.

## 6. Chronological, purged, embargoed splits

Choose split boundary timestamps once from the full regular time grid, before
dropping rows for missing features/targets and before considering any outcome.
Use the same boundaries for all horizons and models. Time-duration boundaries
are preferable to quantiles of filtered row counts because missingness can vary
by regime.

Let the train/validation boundary be `B1`, validation/test boundary be `B2`, and
`M` the maximum enabled prediction horizon. Let
`E >= ceil(M / sampling_interval) * sampling_interval`. A conservative common
mask is:

```text
train:      t < B1       and t + M < B1
validation: t >= B1 + E  and t < B2 and t + M < B2
test:       t >= B2 + E
```

The apparently redundant conditions make the contract explicit:

- **purge** removes source rows whose label endpoint reaches or crosses the next
  segment (`target_end_time >= boundary`);
- **embargo** removes the first `E` of the destination segment.

For horizon-specific masks, `H` can replace `M`, but common `M` masks make model
comparisons use the same timestamps and are harder to misuse. If optional 300 s
is enabled, it becomes part of `M`.

Each expanding walk-forward fold repeats the same logic at its own assessment
boundary. Preprocessing/model fitting occurs in a pipeline fit only on that
fold's training mask. Feature construction itself may be performed once globally
only because the prefix-invariance test proves it is causal.

After model/feature/threshold choices are frozen, a final model may be refit on
all pre-test rows satisfying `t + M < B2`; the final test still begins at
`B2 + E`. Test data may not choose imputation, scaling, features, model,
hyperparameters, signal orientation, quantile cutpoints, regime cutpoints,
extreme-event cutoff, trading threshold, cost scenario, or verdict thresholds.

Minimal split assertions for a 1-second grid `t=0..29`, `M=3`, `E=3`, `B1=10`,
`B2=20`:

- train times are `0..6` because `t+3 < 10`;
- validation times are `13..16` because of the leading embargo and trailing
  purge;
- test starts at `23`;
- `max(train.target_end_time) < 10`;
- `min(validation.time) >= 13` and
  `max(validation.target_end_time) < 20`;
- `min(test.time) >= 23`;
- all three sets are disjoint, sorted, and preserve original order.

Also test non-integer horizons/sampling by timestamp arithmetic, and fail early
when a segment has no rows after purge/embargo rather than weakening the rules.

## 7. Correlation, stability, and metric edge cases

All pairwise statistics use pairwise-complete finite observations and save the
actual `n`. Pearson/Spearman/MI are `NaN` for constant features or targets, not
zero. A near-zero-variance flag should be based on a documented scale-aware
threshold or unique-value fraction and is descriptive, not a silent exclusion.

Fold stability uses folds as the independent reporting units:

```text
mean_IC          = mean(IC_fold)
median_IC        = median(IC_fold)
IC_sd            = sample standard deviation across folds (ddof=1)
IC_t             = mean_IC / (IC_sd / sqrt(K))
fraction_positive = count(IC_fold > 0) / K
fraction_same_sign = count(sign(IC_fold) == sign(IC_full_pretest)) / K
```

`IC_t` is unavailable for fewer than three finite folds or zero fold variance.
Do not substitute the number of rows for `K`. Report worst and best fold without
hiding them in the average. Day/hour statistics require a minimum sample count
and both variables non-constant; sparse groups remain present with `NaN` metrics.

Overlapping `H`-second labels sampled every second create strong serial
dependence and an effective sample size far below the row count. Ordinary
`scipy` Pearson p-values can be saved as `p_iid_approx` for reference, but they
must not drive the verdict.

Classification reports use a fixed label order `[-1,0,+1]`, retaining a 3x3
confusion matrix even when a class is absent. Per-class precision/recall/F1 are
`NaN` when their denominator is undefined (rather than silently claiming zero).
Multiclass Brier score is
`mean_i sum_k (p_ik - 1[y_i=k])^2`. ROC/PR AUC is one-vs-rest per class only when
that class has both positive and negative examples; the macro average states
which classes were included. Calibration is also one-vs-rest and includes bin
counts and prevalence.

## 8. Time-series uncertainty: block bootstrap and HAC

### 8.1 Moving-block bootstrap

For IC, resample **paired, chronologically adjacent** `(signal,target)` rows in
blocks. For strategy results, resample a regularly indexed opportunity/PnL
series in calendar-time blocks so quiet and active clusters remain represented.
Never independently resample signal and target.

A transparent pilot rule is:

```text
H_rows = ceil(H / sampling_interval)
block_length >= H_rows
```

Predefine a primary block length using pre-test data, and report sensitivity at
`H_rows`, `2*H_rows`, and `5*H_rows` where the sample permits. The block must not
cross a recorded data gap, invalid interval, or session boundary; draw only from
contiguous valid runs, concatenate sampled blocks, and truncate to the requested
length. Use at least 1,000 deterministic-seed replicates for routine output.
The 95% percentile CI is bootstrap quantiles `[0.025,0.975]`.

If `n < 5 * block_length`, there are too few effective blocks for a persuasive
CI: still return a clearly flagged exploratory interval or `NaN`, but do not
claim significance. Constant bootstrap replicates are excluded for correlation,
and the retained replicate count is reported.

A paired block bootstrap estimates the sampling interval under the observed
relationship. For a null p-value, use a block-aware null such as large circular
shifts of the entire target relative to all features (same shift for all features
within a replicate), excluding shifts smaller than the dependence block. This
preserves each series' autocorrelation while breaking contemporaneous alignment.

### 8.2 Newey-West/HAC

For mean return, or the slope in `y_t = alpha + beta*x_t + error_t`, use a
Newey-West covariance with Bartlett weights. Given centered scalar series `z_t`:

```text
gamma_l = (1/n) * sum_{t=l+1..n} z_t * z_(t-l)
LRV     = gamma_0 + 2 * sum_{l=1..q} (1 - l/(q+1)) * gamma_l
SE(mean)= sqrt(LRV / n)
t_HAC   = mean(z) / SE(mean)
```

For overlapping horizon returns on a 1-second grid, use at least
`q = H_rows - 1` (Hansen-Hodrick overlap logic); a larger predeclared lag based
on pre-test autocorrelation is acceptable. Pearson significance can be obtained
from the HAC slope after standardizing finite `x` and `y`; Spearman uses ranks
then the same regression. State `q` in every output.

Average **conditional per-trade** returns occur at irregular times, so row-count
HAC can be misleading. Prefer the calendar-time block bootstrap for that metric.
HAC is most defensible on a regular per-opportunity payoff series aligned to each
grid decision (zero when the fixed rule makes no trade), while conditional
average return per trade gets a block-bootstrap CI.

Edge handling:

- `SE=0`, negative LRV from numerical error, `n <= q+2`, or too few finite values
  produces `NaN` plus a reason, never infinite significance.
- No annualization is applied to a several-hour smoke sample.
- The bootstrap/HAC unit must match the reported estimand: per opportunity,
  per non-overlapping trade, or per unit clock time. Never switch denominators.

Minimal statistical tests:

1. Same seed and input produce identical bootstrap endpoints.
2. A constant signal returns `NaN` IC without crashing.
3. Encoded row IDs show that rows inside every sampled block are consecutive and
   no block crosses a declared gap.
4. Perfectly increasing paired series has IC 1 in the original sample; its block
   CI remains within `[-1,1]`.
5. Constant return series gives zero variance and `NaN` Sharpe/t-stat rather than
   infinity.

## 9. Quantile analysis

Quantile tables must use out-of-sample predictions. Training-set fitted values
are never acceptable evidence. Standalone feature direction/orientation is fixed
from train/validation only and stored; it cannot be flipped after seeing test.

For confirmatory test tables, learn quantile cutpoints from validation signals
and freeze them before applying to test. Within-fold equal-frequency bins based
only on that fold's OOS signal are useful descriptive diagnostics, but should be
labeled transductive/descriptive because the full fold's signal distribution is
used.

Do not split tied values arbitrarily to force ten equal buckets. Collapse duplicate
edges and report `requested_bucket_count`, `actual_bucket_count`, edge values,
and empty buckets. A constant or highly discrete signal may legitimately produce
one bucket; that is evidence against useful ranking, not an exception to hide.

For each bucket save:

- `n`, mean, median, sample standard deviation, and optionally block-bootstrap CI
  of the future return;
- positive-return rate `P(r>0)` (hypothetical long-direction accuracy);
- negative-return rate `P(r<0)` (hypothetical short-direction accuracy);
- flat/dead-zone class rate and long/short class rates;
- model sign hit `P(sign(prediction)==sign(return))` only when prediction has a
  defined return direction;
- prescribed-strategy hit only when a threshold/action mapping was fixed on
  validation. A raw feature bucket has no honest generic "hit rate" until such a
  mapping exists.

Zeros remain in the denominator and are neither long nor short wins. Save both
return-sign rates and dead-zone class rates so the definition is not ambiguous.

Useful monotonicity diagnostics are descriptive, not a replacement for the raw
curve:

```text
extreme_spread = mean_return(top_bucket) - mean_return(bottom_bucket)
adjacent_monotone_fraction = count(mean_(j+1) >= mean_j) / (B-1)
bucket_rank_rho = Spearman(bucket_index, bucket_mean_return)
```

For a direction expected to be negative, orient on pre-test data or reverse the
inequality, but preserve the raw ascending-signal table. Always plot uncertainty
and bucket counts; a monotone curve driven by two tiny extreme buckets is not
robust.

Minimal quantile tests:

- Signal and target `1..100` create ten buckets of ten with strictly increasing
  means, positive extreme spread, and monotone fraction 1.
- A constant signal creates one tie-preserving bucket, not ten arbitrary buckets.
- Validation cutpoints are unchanged when extreme test signals are appended;
  those test values fall into the outer buckets.
- Empty buckets and a bucket with one row report `NaN` sample standard deviation
  without crashing or disappearing from the schema.

## 10. Multiple-testing contract

Predeclare statistical families before reading final-test outcomes. A defensible
minimal family is all finite confirmatory feature-by-required-horizon p-values
for Pearson and Spearman together. If Pearson, Spearman, and MI are treated as
separate families instead, say so prominently because it weakens the global
error guarantee. Day/hour/fold slices are exploratory families and must not be
mined for a headline claim.

Primary raw p-values should be serial-dependence-aware (HAC slope or shared
circular-shift null). Mutual information has no trustworthy analytic p-value in
this setting; use the block/circular-shift null or leave p/q blank. Do not feed
naive IID p-values into the headline FDR table as though they were valid.

Benjamini-Hochberg for `m` finite p-values:

1. sort `p_(1) <= ... <= p_(m)`;
2. compute `v_i = m * p_(i) / i`;
3. enforce monotonicity backward,
   `q_(i) = min(1, min_{j>=i} v_j)`;
4. map q-values back to original order; exclude `NaN` from `m` but preserve it
   as `NaN` in output;
5. `reject_fdr = q <= configured_alpha`.

Known toy: input p-values `[0.01, 0.04, 0.03, 0.002, NaN]` yield BH q-values
`[0.02, 0.04, 0.04, 0.008, NaN]` when `m=4`.

Save family name, family size, method, alpha, raw p, q, rejection flag, horizon,
feature/model, metric, fold/stage, and the p-value method/lag/block length. Apply
BH separately to pre-test discovery and final confirmation; never let many
exploratory fold/hour tests dilute or redefine the confirmatory family.

Testing several models, horizons, thresholds, costs, ablations, and regimes also
creates researcher degrees of freedom even if there is no p-value column. The
final test should report all predeclared candidates, not select the best as the
only headline. A GO requires coherent fold, quantile, concentration, ablation,
and economic evidence; `q<0.05` alone is insufficient.

## 11. Execution and PnL sign contract

The classification dead zone, trading threshold, and transaction-cost scenario
are separate objects. Let the frozen validation-selected action be

```text
s_t = +1 if prediction_t > +theta
      -1 if prediction_t < -theta
       0 otherwise
```

Equality to the threshold is flat. Costs are charged only when `s_t != 0` and
only once per completed round trip.

For horizon `H`, optimistic gross log return is

```text
gross_mid_log = s_t * log(mid_(t+H) / mid_t)
```

The taker approximation already includes crossing the visible spread:

```text
long:  gross_taker_log = log(bid_(t+H) / ask_t)
short: gross_taker_log = log(bid_t / ask_(t+H))
flat:  0
```

Use quotes known at the exact entry/exit grid timestamps. Entry at the same tick
as feature observation assumes zero decision/network latency and must be labeled
optimistic even in the "taker" table. Missing/stale entry or exit quotes make the
trade unpriceable; count it explicitly. Do not silently move the exit to the next
valid quote, and do not use future exit availability as a signal-selection rule.

To avoid cost ambiguity, define each simple sensitivity value as
`round_trip_incremental_cost_bps`. Then

```text
net_log_bps = 10,000 * gross_log - round_trip_incremental_cost_bps
```

For the mid diagnostic, this scenario may represent all assumed round-trip
execution costs. For the taker diagnostic, spread is already in bid/ask prices,
so the incremental scenario must represent only fees/slippage not already in the
quotes. If fees/slippage are configured per side instead, convert and display
their two-sided total; never subtract the spread twice. These are log-return-bps
diagnostics, not an assertion about an exchange's actual fee schedule.

Default to non-overlapping positions for the pilot: enter only while flat, hold
exactly `H`, ignore intervening signals, then exit. This makes exposure bounded by
one unit and cumulative wealth interpretable. If every 1-second signal is treated
as an independent overlapping bet instead, label it explicitly, account for
simultaneous gross exposure/capital, and do not present an unscaled cumulative
sum as a realizable equity curve.

For a non-overlapping log-return series ordered by exit time:

```text
equity_j = exp(cumulative_sum(net_log_j))
drawdown_j = equity_j / cumulative_max(equity_j) - 1
max_drawdown = min(drawdown_j)
profit_factor = sum(positive net returns) / abs(sum(negative net returns))
```

Profit factor is `NaN` for no trades, `0` for losses with no wins, and `+inf`
with an explicit flag for wins with no losses. Report an unannualized per-trade
Sharpe-like ratio `mean(return)/sample_sd(return)` and Sortino
`mean(return)/sqrt(mean(min(return,0)^2))`; undefined denominators produce
`NaN`. Do not annualize a short smoke run. Also report raw one-way turnover
`sum(abs(position_t-position_(t-1)))`, round trips, turnover per hour/day,
time-weighted `mean(abs(position_t))` exposure, and actual holding seconds.

Minimal PnL sign tests:

1. Mid 100 to 110: long is `+log(1.1)`, short is `-log(1.1)`.
2. Entry bid/ask `99/101`, exit `109/111`: taker long is
   `log(109/101)>0`; taker short is `log(99/111)<0`.
3. Entry `99/101`, exit `89/91`: long is `log(89/101)<0`; short is
   `log(99/91)>0`.
4. Unchanged `99/101`: optimistic PnL is zero, but both taker directions lose
   `log(99/101)` before incremental fees.
5. Net PnL is never greater than gross PnL for a non-negative cost; flat rows
   remain exactly zero and incur no cost.
6. With two completed non-overlapping trades, raw one-way turnover is four
   position changes (two entries, two exits), absent a direct reversal.

## 12. Data-quality definitions worth testing

- Count WebSocket envelopes separately from contained trade records/book
  updates. A public-trade message can contain multiple trades.
- Session duration is last minus first local receive timestamp. For an empty or
  one-message session, spacing/frequency statistics are `NaN`.
- Event spacing is calculated per topic/stream from consecutive local receive
  times; control messages are excluded. Report median and p90/p95/p99 spacing and
  a separately named average event rate. `1/median_spacing` is not a median
  frequency distribution.
- Latency is `local_receive_wall_time - exchange_timestamp`. Negative latency is
  possible when the collector clock is skewed and must be reported, not clipped.
  A process-monotonic clock handles ordering/staleness but cannot by itself
  estimate exchange-to-local latency.
- Missing/stale intervals are contiguous runs on the complete expected grid.
  State whether boundary partial intervals count. Longest gap is elapsed clock
  duration, not retained-row count.
- Report both `book_usable_pct` (valid, non-stale book / expected grid rows) and
  `model_usable_pct_H` (all required features and target / expected rows). Their
  denominators must include invalid grid rows, otherwise dropping bad periods can
  misleadingly produce 100%.
- Duplicate trade IDs are removed from features but retained/countable in raw
  diagnostics. Duplicate/order anomaly counters are per session and total.

## 13. Minimal test suite map

The following is enough coverage without building a large testing framework:

| Test area | Essential assertions |
|---|---|
| `test_book` | full replacement, insert/update/delete, atomicity, duplicate/conflict, `u` gap, invalid-until-snapshot, reconnect, crossed/empty state |
| `test_features` | microprice/OBI known values and bounds, zero denominators, trade-side signs, right-closed windows, OFI table above, prefix invariance |
| `test_targets` | exact timestamp pairing, missing-row protection, endpoint validity, dead-zone boundaries |
| `test_splits` | toy boundary sets above, purge label endpoints, embargo, ordering/disjointness, empty-split failure |
| `test_preprocessing` | scaler/imputer fit-call sees training indices only; appending/changing test values cannot alter fitted parameters or validation predictions |
| `test_quantiles` | monotone 1..100 case, tie collapse, frozen validation edges, empty/singleton bucket schema |
| `test_statistics` | BH toy, seeded block bootstrap, block gap integrity, constant-series behavior, CI/correlation bounds |
| `test_trading` | all long/short sign cases, cost exactly once, no cost while flat, non-overlap, equity/drawdown, undefined metric handling |
| `test_pipeline_toy` | one deterministic synthetic end-to-end table produces the expected rows, folds, predictions schema, raw output tables, and no final-test access during selection |

The end-to-end synthetic dataset should deliberately include one duplicate delta,
one delete, one gap/resnapshot, one no-trade window, one missing grid tick, a
future price jump, and both rising/falling price paths. Its assertions should be
on exact row timestamps and a handful of exact numeric outputs, not merely
"pipeline did not crash."

## 14. Verification order and stop conditions

Run checks in this order so modeling cannot beautify broken inputs:

1. deterministic unit tests;
2. raw schema and parse audit;
3. book state-machine replay and validity intervals;
4. complete-grid/data-quality report;
5. feature prefix-invariance and target endpoint audit;
6. split/purge/embargo manifest audit;
7. fold-local preprocessing/model run;
8. OOS metrics, quantiles, uncertainty, and multiple-testing outputs;
9. frozen-threshold execution diagnostics;
10. final-test read exactly once by the analysis stage.

Hard stop before modeling if the book cannot be reconstructed, split masks are
empty, timestamps are not monotonic within the declared ordering key, the
processed grid contains duplicate timestamps, or causal/prefix-invariance tests
fail. Low usable percentage, wide latency, gaps, class imbalance, insufficient
effective blocks, or sparse trades need not crash the run, but they must force a
prominent `INSUFFICIENT_DATA`/caveat outcome rather than a GO.

The smoke collection only verifies plumbing. No IC, p-value, quantile curve, or
PnL from a several-minute smoke sample is scientific evidence of alpha.
