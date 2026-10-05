# Leakage checklist

This file is a manual audit contract. A run is not scientifically valid unless
every applicable item is true and the run manifest contains the evidence.

- [ ] The complete regular UTC grid was built before any row filtering.
- [ ] A state at grid time `t` uses only frames with local receive time `<= t`.
- [ ] All joins of trades/ticker state are backward/as-of with a reported age or
      exact trailing interval `(t-W, t]`.
- [ ] No centered rolling window, future interpolation or backward fill exists.
- [ ] Order-book features are `NaN` while unsynchronised/stale and no OFI window
      crosses a snapshot, reconnect, transport loss or invalid interval.
- [ ] Targets use the exact grid timestamp `t+H`; missing rows never shorten a
      horizon.
- [ ] Invalid target endpoints and invalid target paths are explicit flags.
- [ ] Train/validation/test boundaries were frozen from timestamps before
      inspecting target/model results and are shared by all models/horizons.
- [ ] Every temporal boundary uses purge and an embargo at least as long as the
      largest enabled horizon.
- [ ] Walk-forward folds expand chronologically; observations are never randomly
      split or shuffled between folds.
- [ ] Imputers/scalers are fitted only on each fold's training slice.
- [ ] Feature orientation, feature groups, model grid, quantile/regime edges,
      extreme-event cutoff and trading threshold are selected on development
      data only.
- [ ] The final test does not select features, model, hyperparameters, threshold,
      cost scenario or verdict thresholds and is evaluated once at the end.
- [ ] Model matrices reject all `target_*`, `future_*`, timestamp/split and output
      columns through an explicit allow-list.
- [ ] Prefix-invariance tests prove that appending future events cannot change
      already-built features.
- [ ] Feature/target statistics use pairwise-complete observations; missing data
      are not globally replaced by zero.
- [ ] IID p-values, if present, are labelled approximate; confirmatory inference
      uses HAC or paired time-series blocks and Benjamini-Hochberg families are
      predeclared.
- [ ] Raw data are immutable and the run stores config, seed, timestamps, feature
      list, boundaries, environment versions, source commit when available and a
      full dataset fingerprint.
- [ ] A short smoke run is labelled `INSUFFICIENT DATA — SMOKE ONLY`, never GO.

