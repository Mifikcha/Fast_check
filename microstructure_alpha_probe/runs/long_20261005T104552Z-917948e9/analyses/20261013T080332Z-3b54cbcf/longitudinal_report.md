# Stage 2 longitudinal robustness report

Experiment: `20261005T104552Z-917948e9`; 7 complete UTC days. Partial smoke days, if any, are not used in verdicts.

## A. Frozen Ridge@30s replication

Original 24h rank IC: 0.1891. New mean daily rank IC: 0.0607; median 0.0462; positive fraction 1.00. Day-level 95% t interval: [0.0301, 0.0912] (unavailable for fewer than 3 days).

## B. Non-overlapping labels

Mean phase IC 0.0607; minimum 0.0501; positive phase fraction 1.00. All 30 UTC phase offsets are included.

## C. Latency

`latency_stress.csv` separates stale raw-book OBI from delayed execution of the frozen Ridge signal. The clock is local receive time; first valid raw quote after the deadline is used (maximum 250 ms wait).

## D. Longer horizons

10/30/60/120/300/900 s are exploratory applications of the frozen 30 s signal, not retrained forecasts.

## E. OBI alone

Direct, untrained OBI L1/L5/L10/L20 and microprice deviation results are in `obi_baselines.csv`.

## F. Incremental order-book information

Mean order-book IC minus price-history IC: 0.1311. Ablation models were trained on the original development period only.

## G. Non-zero costs

Frozen 30 s net 1 bps total PnL (log units): -1.24701; trades 15624. Costs are hypothetical round-trip increments on top of observed bid/ask execution.

## H. Days, regimes, and quality

Global buffered usability: 99.73%. Local gaps exclude affected samples and their ±60 s neighborhood. Regime results are descriptive, not selection rules.

## I. Verdicts

Scientific: **GO**. Trading: **NO-GO**.

Daily rows are the inference unit. Observation count is not treated as independent evidence; small-day confidence intervals are explicitly limited. See all CSVs and plots for auditability.
