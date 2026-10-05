"""Immutable longitudinal aggregation of completed Stage 2 day artifacts."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from .config import project_path
from .io_utils import atomic_write_dataframe, atomic_write_json, atomic_write_text
from .long_collection import recording_sessions
from .long_pipeline import TABLE_NAMES, run_daily
from .statistics import bootstrap_correlation


def _number(value: Any) -> float:
    try:
        result = float(value)
        return result if np.isfinite(result) else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def _fmt(value: Any, digits: int = 4) -> str:
    number = _number(value)
    return f"{number:.{digits}f}" if np.isfinite(number) else "n/a"


def _stats(values: Any) -> dict[str, Any]:
    x = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(float)
    n = len(x)
    low, high = (stats.t.interval(.95, n-1, loc=x.mean(), scale=stats.sem(x))
                 if n >= 3 and np.std(x, ddof=1) > 0 else (np.nan, np.nan))
    return {"n": n, "mean": float(x.mean()) if n else np.nan,
            "median": float(np.median(x)) if n else np.nan,
            "std": float(np.std(x, ddof=1)) if n > 1 else np.nan,
            "min": float(x.min()) if n else np.nan, "max": float(x.max()) if n else np.nan,
            "positive_fraction": float(np.mean(x > 0)) if n else np.nan,
            "ci95_low": float(low), "ci95_high": float(high)}


def _plot(path: Path, title: str, series: list[tuple[Any, Any, str]], *, kind: str = "line",
          ylabel: str = "") -> None:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for x, y, label in series:
        if len(x):
            (ax.bar if kind == "bar" else ax.plot)(x, y, label=label)
    ax.axhline(0, color="0.45", lw=.8)
    ax.set(title=title, ylabel=ylabel)
    if len(series) > 1:
        ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def render_long_report(analysis: Path) -> Path:
    """Regenerate the Markdown and 16 figures from saved tables; no model refit."""
    tables = {name: pd.read_csv(analysis / name) for name in TABLE_NAMES}
    summary = json.loads((analysis / "longitudinal_summary.json").read_text(encoding="utf-8"))
    daily = tables["daily_confirmatory_metrics.csv"].sort_values("day")
    phase = tables["nonoverlap_ic_by_phase.csv"]
    latency = tables["latency_stress.csv"]
    horizon = tables["horizon_comparison.csv"]
    cost = tables["cost_sensitivity.csv"]
    select = tables["selectivity_analysis.csv"]
    obi = tables["obi_baselines.csv"]
    group = tables["feature_group_by_day.csv"]
    quality = tables["data_quality_by_day.csv"]
    quantile = tables["quantile_by_day.csv"]
    figs = analysis / "plots"
    figs.mkdir(exist_ok=True)
    days = daily["day"].astype(str).to_list()
    ic = pd.to_numeric(daily["rank_ic"], errors="coerce")
    _plot(figs / "daily_ic.png", "Frozen Ridge@30s: daily rank IC", [(days, ic, "new days")], kind="bar", ylabel="rank IC")
    _plot(figs / "expanding_ic.png", "Expanding mean daily IC", [(days, ic.expanding().mean(), "daily mean")], ylabel="rank IC")
    _plot(figs / "daily_ic_distribution.png", "Distribution of daily IC", [(list(range(len(ic))), np.sort(ic), "sorted days")], ylabel="rank IC")
    _plot(figs / "original_vs_new.png", "Original 24h vs independent days", [(days, ic, "new days"), (days, [summary["original_rank_ic"]]*len(days), "original 24h")], ylabel="rank IC")
    ph = phase.groupby("phase", as_index=False)["spearman"].mean() if len(phase) else phase
    _plot(figs / "nonoverlap_ic_by_phase.png", "Every non-overlapping 30s phase", [(ph.get("phase", []), ph.get("spearman", []), "phase")], kind="bar", ylabel="rank IC")
    for mode, signal in (("stale_obi1", "stale OBI"), ("delayed_ridge30", "delayed Ridge")):
        sub = latency.loc[latency["mode"] == mode].groupby("latency_ms", as_index=False).mean(numeric_only=True) if len(latency) else latency
        if mode == "stale_obi1":
            stale = sub
        else:
            delayed = sub
    _plot(figs / "ic_vs_latency.png", "Predictive IC vs latency", [(x.get("latency_ms", []), x.get("rank_ic", []), label) for x, label in ((stale, "stale OBI"), (delayed, "delayed Ridge"))], ylabel="rank IC")
    _plot(figs / "pnl_vs_latency.png", "Executable edge vs latency", [(x.get("latency_ms", []), x.get("average_taker_edge_bps", []), label) for x, label in ((stale, "stale OBI"), (delayed, "delayed Ridge"))], ylabel="bps/trade")
    h = horizon.groupby("horizon_s", as_index=False).mean(numeric_only=True) if len(horizon) else horizon
    _plot(figs / "ic_vs_horizon.png", "Frozen signal vs horizon", [(h.get("horizon_s", []), h.get("rank_ic", []), "Ridge@30s")], ylabel="rank IC")
    _plot(figs / "extreme_spread_vs_horizon.png", "Extreme decile spread vs horizon", [(h.get("horizon_s", []), h.get("extreme_bucket_spread_bps", []), "Ridge")], ylabel="bps")
    b = tables["breakeven_costs.csv"].groupby("horizon_s", as_index=False).mean(numeric_only=True) if len(horizon) else tables["breakeven_costs.csv"]
    _plot(figs / "breakeven_cost_vs_horizon.png", "Breakeven round-trip cost", [(b.get("horizon_s", []), b.get("breakeven_cost_bps", []), "Ridge")], ylabel="bps")
    o = obi.loc[(obi["horizon_s"] == 30) & (obi["signal"] == "obi_1")]
    _plot(figs / "obi_ic_by_day.png", "OBI L1 rank IC by day", [(o.get("day", []), o.get("rank_ic", []), "OBI L1")], kind="bar", ylabel="rank IC")
    q = quantile.loc[quantile["signal"] == "prediction"]
    qmatrix = q.pivot_table(index="day", columns="bucket", values="mean_return_bps", aggfunc="first").reindex(columns=range(1, 11)) if len(q) else pd.DataFrame(columns=range(1, 11))
    fig, ax = plt.subplots(figsize=(9, max(3, .45 * max(1, len(qmatrix)) + 2)))
    image = ax.imshow(qmatrix.to_numpy(float) if len(qmatrix) else np.full((1, 10), np.nan), aspect="auto", cmap="coolwarm")
    fig.colorbar(image, ax=ax, label="mean future return (bps)")
    ax.set(xticks=range(10), xticklabels=[f"Q{i}" for i in range(1, 11)],
           yticks=range(max(1, len(qmatrix))), yticklabels=qmatrix.index.to_list() or ["no data"],
           title="Frozen Ridge decile mean return by UTC day (bps)")
    fig.tight_layout()
    fig.savefig(figs / "quantile_returns_by_day.png", dpi=150)
    plt.close(fig)
    _plot(figs / "orderbook_vs_price_history_ic.png", "Incremental book information by day", [(x.get("day", []), x.get("rank_ic", []), label) for x, label in ((group.loc[group["feature_group"] == "order_book"], "order book"), (group.loc[group["feature_group"] == "price_history"], "price history"))], ylabel="rank IC")
    c = cost.loc[cost["horizon_s"] == 30].groupby("cost_bps", as_index=False).mean(numeric_only=True) if len(cost) else cost
    _plot(figs / "cost_sensitivity.png", "30s net edge under hypothetical costs", [(c.get("cost_bps", []), c.get("net_edge_bps", []), "net")], ylabel="bps/trade")
    s = select.loc[select["horizon_s"] == 30].groupby("tail_percent_each_side", as_index=False).mean(numeric_only=True) if len(select) else select
    _plot(figs / "selectivity_edge.png", "Frozen selectivity thresholds vs edge", [(s.get("tail_percent_each_side", []), s.get("executable_edge_bps", []), "executable")], ylabel="bps/trade")
    _plot(figs / "data_usability_by_day.png", "Buffered data usability by day", [(quality.get("day", []), quality.get("usable_pct", []), "usable")], kind="bar", ylabel="percent")
    lines = ["# Stage 2 longitudinal robustness report", "", f"Experiment: `{summary['experiment_id']}`; {summary['complete_days']} complete UTC days. Partial smoke days, if any, are not used in verdicts.", "",
             "## A. Frozen Ridge@30s replication", "", f"Original 24h rank IC: {_fmt(summary['original_rank_ic'])}. New mean daily rank IC: {_fmt(summary['daily_ic']['mean'])}; median {_fmt(summary['daily_ic']['median'])}; positive fraction {_fmt(summary['daily_ic']['positive_fraction'], 2)}. Day-level 95% t interval: [{_fmt(summary['daily_ic']['ci95_low'])}, {_fmt(summary['daily_ic']['ci95_high'])}] (unavailable for fewer than 3 days).", "",
             "## B. Non-overlapping labels", "", f"Mean phase IC {_fmt(summary['phase_ic']['mean'])}; minimum {_fmt(summary['phase_ic']['min'])}; positive phase fraction {_fmt(summary['phase_ic']['positive_fraction'], 2)}. All 30 UTC phase offsets are included.", "",
             "## C. Latency", "", "`latency_stress.csv` separates stale raw-book OBI from delayed execution of the frozen Ridge signal. The clock is local receive time; first valid raw quote after the deadline is used (maximum 250 ms wait).", "",
             "## D. Longer horizons", "", "10/30/60/120/300/900 s are exploratory applications of the frozen 30 s signal, not retrained forecasts.", "",
             "## E. OBI alone", "", "Direct, untrained OBI L1/L5/L10/L20 and microprice deviation results are in `obi_baselines.csv`.", "",
             "## F. Incremental order-book information", "", f"Mean order-book IC minus price-history IC: {_fmt(summary['group_ic_lift'])}. Ablation models were trained on the original development period only.", "",
             "## G. Non-zero costs", "", f"Frozen 30 s net 1 bps total PnL (log units): {_fmt(summary['net_1bps_pnl_log'], 5)}; trades {summary['trades_30s']}. Costs are hypothetical round-trip increments on top of observed bid/ask execution.", "",
             "## H. Days, regimes, and quality", "", f"Global buffered usability: {_fmt(summary['global_usable_pct'], 2)}%. Local gaps exclude affected samples and their ±60 s neighborhood. Regime results are descriptive, not selection rules.", "",
             "## I. Verdicts", "", f"Scientific: **{summary['scientific_verdict']}**. Trading: **{summary['trading_verdict']}**.", "", "Daily rows are the inference unit. Observation count is not treated as independent evidence; small-day confidence intervals are explicitly limited. See all CSVs and plots for auditability."]
    target = analysis / "longitudinal_report.md"
    if target.exists():
        target.unlink()
    atomic_write_text("\n".join(lines) + "\n", target)
    return target


def analyze_long(config: dict[str, Any], manifest: Path, *, include_partial: bool = False,
                 progress: Callable[[dict[str, Any]], None] | None = None) -> Path:
    days = run_daily(config, manifest, include_partial=include_partial, progress=progress)
    root = project_path(config, "runs_dir") / f"long_{manifest.parent.name}"
    output = root / "analyses" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8])
    output.mkdir(parents=True, exist_ok=False)
    complete = [p for p in days if json.loads((p / "daily_manifest.json").read_text(encoding="utf-8"))["complete_utc_day"]]
    selected = complete or days  # Partial data is visible for smoke, never counted as replication.
    tables = {}
    for name in TABLE_NAMES:
        pieces = [pd.read_csv(day / name) for day in selected]
        table = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()
        tables[name] = table
        atomic_write_dataframe(table, output / name)
    sessions = recording_sessions(manifest, project_path(config, "raw_dir"))
    atomic_write_dataframe(sessions, output / "recording_sessions.csv")
    daily = tables["daily_confirmatory_metrics.csv"]
    phase = tables["nonoverlap_ic_by_phase.csv"]
    group = tables["feature_group_by_day.csv"]
    quality = tables["data_quality_by_day.csv"]
    cost = tables["cost_sensitivity.csv"]
    quantile = tables["quantile_by_day.csv"]
    matrix = (quantile.loc[quantile["signal"] == "prediction"]
              .pivot_table(index="day", columns="bucket_name", values="mean_return_bps", aggfunc="first")
              .reindex(columns=[f"Q{i}" for i in range(1, 11)])) if len(quantile) else pd.DataFrame(columns=[f"Q{i}" for i in range(1, 11)])
    atomic_write_dataframe(matrix.reset_index(), output / "quantile_day_matrix.csv")
    import yaml
    frozen = yaml.safe_load((Path(__file__).parents[1] / "configs" / "confirmatory_ridge30.yaml").read_text(encoding="utf-8"))
    cfg = frozen["stage2"]
    daily_ic = _stats(daily["rank_ic"])
    daily_ic["same_sign_as_original_fraction"] = float(np.mean(np.sign(daily["rank_ic"]) == np.sign(frozen["source_test_rank_ic"]))) if len(daily) else np.nan
    phase_mean = phase.groupby("phase")["spearman"].mean() if len(phase) else pd.Series(dtype=float)
    phase_ic = _stats(phase_mean)
    group_means = group.pivot_table(index="day", columns="feature_group", values="rank_ic") if len(group) else pd.DataFrame()
    lift = float((group_means["order_book"] - group_means["price_history"]).mean()) if {"order_book", "price_history"}.issubset(group_means) else np.nan
    confirm_cost = cost.loc[(cost["horizon_s"] == 30) & (cost["cost_bps"] == cfg["trading_verdict"]["confirmatory_cost_bps"])]
    net = float(confirm_cost["net_taker_pnl_log"].sum())
    trades = int(confirm_cost["trades"].sum())
    usable = 100 * quality["final_usable_duration_s"].sum() / quality["raw_duration_s"].sum() if len(quality) else np.nan
    n_complete = len(complete)
    enough = (n_complete >= int(cfg["minimum_complete_days"])
              and usable >= cfg["minimum_global_usable_pct"]
              and bool((daily["n"] >= cfg["minimum_daily_rows"]).all()))
    v = cfg["scientific_verdict"]
    if not enough:
        science = "INSUFFICIENT EVIDENCE"
    elif daily_ic["mean"] <= 0 or daily_ic["positive_fraction"] < .5:
        science = "NO-GO"
    elif (daily_ic["mean"] >= v["minimum_daily_mean_rank_ic"] and
          daily_ic["positive_fraction"] >= v["minimum_positive_day_fraction"] and
          phase_ic["positive_fraction"] >= v["minimum_positive_phase_fraction"] and
          lift > v["minimum_group_ic_lift"]):
        science = "STRONG GO" if n_complete >= v["strong_minimum_days"] and daily_ic["ci95_low"] > 0 else "GO"
    else:
        science = "WEAK GO"
    trading = "INSUFFICIENT EVIDENCE" if not enough else ("GO" if trades >= cfg["trading_verdict"]["minimum_confirmatory_trades"] and net > 0 else "NO-GO")
    sampled = []
    for day_path in complete:
        predictions = pd.read_parquet(day_path / "predictions.parquet")
        ts = pd.to_datetime(predictions["timestamp"], utc=True)
        phase_zero = ((ts.astype("int64") // 1_000_000_000) % 30 == 0) & predictions["eligible_30s"]
        sampled.append(predictions.loc[phase_zero, ["prediction", "target_return_30s"]].assign(day=day_path.name))
    block = pd.concat(sampled, ignore_index=True) if sampled else pd.DataFrame()
    block_ci = (bootstrap_correlation(block["prediction"], block["target_return_30s"],
                                      block_length=10, n_resamples=200, segment_ids=block["day"], seed=10072)
                if len(block) >= 10 else {"reason": "too_few_nonoverlap_samples", "n": len(block)})
    summary = {"experiment_id": manifest.parent.name, "complete_days": n_complete,
               "partial_smoke_days": len(days)-n_complete, "original_rank_ic": frozen["source_test_rank_ic"],
               "daily_ic": daily_ic, "phase_ic": phase_ic, "group_ic_lift": lift,
               "net_1bps_pnl_log": net, "trades_30s": trades,
               "global_usable_pct": usable, "scientific_verdict": science, "trading_verdict": trading,
               "nonoverlap_phase0_block_bootstrap": block_ci,
               "expanding_daily_mean_ic": [{"day": str(day), "mean_ic": float(x)} for day, x in zip(daily["day"], daily["rank_ic"].expanding().mean())],
               "feature_groups": {name: _stats(values) for name, values in group.groupby("feature_group")["rank_ic"]} if len(group) else {},
               "obi_30s": {name: _stats(values) for name, values in tables["obi_baselines.csv"].loc[tables["obi_baselines.csv"]["horizon_s"] == 30].groupby("signal")["rank_ic"]} if len(tables["obi_baselines.csv"]) else {}}
    atomic_write_json(_json_safe(summary), output / "longitudinal_summary.json")
    render_long_report(output)
    return output
