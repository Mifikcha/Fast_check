from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml


PLOT_NAMES = [
    "01_target_return_distributions.png",
    "02_spread_distribution.png",
    "03_trade_intensity_over_time.png",
    "04_feature_correlation_heatmap.png",
    "05_feature_target_ic_heatmap.png",
    "06_ic_stability_across_folds.png",
    "07_rolling_ic.png",
    "08_prediction_vs_realized.png",
    "09_quantile_returns.png",
    "10_cumulative_gross_pnl.png",
    "11_cumulative_net_pnl.png",
    "12_drawdown.png",
    "13_cost_sensitivity.png",
    "14_model_comparison_ic.png",
    "15_model_comparison_net_pnl.png",
    "16_feature_group_ablation.png",
    "17_permutation_importance.png",
    "18_confusion_matrices.png",
    "19_regime_performance.png",
    "20_time_contribution.png",
]


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or not path.stat().st_size:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _read_json(path: Path, default: Any) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def _replace_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _fmt(value: Any, digits: int = 4) -> str:
    try:
        number = float(value)
        return "n/a" if not np.isfinite(number) else f"{number:.{digits}f}"
    except (TypeError, ValueError):
        return "n/a" if value is None else str(value)


def _placeholder(path: Path, title: str, reason: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.axis("off")
    ax.text(0.5, 0.58, title, ha="center", va="center", fontsize=16, weight="bold")
    ax.text(0.5, 0.43, reason, ha="center", va="center", fontsize=10, wrap=True)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def _plot_guard(path: Path, title: str, fn: Callable[[], None]) -> None:
    try:
        fn()
        if not path.exists():
            _placeholder(path, title, "Plot function produced no output.")
    except Exception as exc:  # plotting must not erase the numerical run
        plt.close("all")
        _placeholder(path, title, f"Unavailable for this run: {type(exc).__name__}: {exc}")


def generate_plots(
    run_dir: str | Path,
    research: pd.DataFrame,
    *,
    feature_columns: list[str],
    primary_model: str,
    primary_horizon_s: int,
    confirmatory_cost_bps: float,
    dpi: int = 140,
) -> list[Path]:
    run = Path(run_dir)
    plots = run / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    fold = _read_csv(run / "fold_metrics.csv")
    corr = _read_csv(run / "feature_target_correlations.csv")
    quantiles = _read_csv(run / "quantile_analysis.csv")
    trading = _read_csv(run / "trading_metrics.csv")
    costs = _read_csv(run / "cost_sensitivity.csv")
    ablation = _read_csv(run / "ablation_metrics.csv")
    importance = _read_csv(run / "permutation_importance.csv")
    classification = _read_csv(run / "classification_metrics.csv")
    regimes = _read_csv(run / "regime_metrics.csv")
    contribution = _read_csv(run / "contribution_metrics.csv")
    predictions_path = run / "predictions_test.parquet"
    trades_path = run / "trades_test.parquet"
    predictions = pd.read_parquet(predictions_path) if predictions_path.exists() else pd.DataFrame()
    trades = pd.read_parquet(trades_path) if trades_path.exists() else pd.DataFrame()

    paths = [plots / name for name in PLOT_NAMES]

    def target_distributions() -> None:
        cols = [c for c in research if c.startswith("target_return_")]
        if not cols:
            raise ValueError("no target columns")
        fig, axes = plt.subplots(len(cols), 1, figsize=(10, 3.2 * len(cols)), squeeze=False)
        for ax, col in zip(axes[:, 0], cols):
            values = research[col].dropna() * 10_000
            if values.empty:
                continue
            sns.histplot(values.clip(values.quantile(0.001), values.quantile(0.999)), bins=80, ax=ax)
            ax.set_title(col)
            ax.set_xlabel("future log return (bps; 0.1%-99.9% view)")
        fig.tight_layout(); fig.savefig(paths[0], dpi=dpi); plt.close(fig)

    def spread_distribution() -> None:
        values = research["spread_bps"].dropna()
        if values.empty:
            raise ValueError("no valid spreads")
        fig, ax = plt.subplots(figsize=(10, 5.5)); sns.histplot(values.clip(upper=values.quantile(0.995)), bins=80, ax=ax)
        ax.set(title="Quoted spread distribution", xlabel="spread (bps)"); fig.tight_layout(); fig.savefig(paths[1], dpi=dpi); plt.close(fig)

    def trade_intensity() -> None:
        col = "trade_count_1s"
        if col not in research:
            raise ValueError(f"missing {col}")
        series = research.set_index("timestamp")[col].rolling(60, min_periods=1).sum()
        fig, ax = plt.subplots(figsize=(12, 5)); ax.plot(pd.to_datetime(series.index, utc=True), series, linewidth=0.8)
        ax.set(title="Trade intensity", ylabel="trades / trailing minute"); fig.tight_layout(); fig.savefig(paths[2], dpi=dpi); plt.close(fig)

    def feature_heatmap() -> None:
        cols = [c for c in feature_columns if c in research][:35]
        if len(cols) < 2:
            raise ValueError("fewer than two features")
        matrix = research[cols].corr(method="spearman")
        fig, ax = plt.subplots(figsize=(max(10, len(cols) * .38), max(8, len(cols) * .34)))
        sns.heatmap(matrix, cmap="vlag", center=0, vmin=-1, vmax=1, ax=ax); ax.set_title("Feature Spearman correlation")
        fig.tight_layout(); fig.savefig(paths[3], dpi=dpi); plt.close(fig)

    def ic_heatmap() -> None:
        subset = corr.copy()
        if "scope" in subset:
            subset = subset[subset["scope"].eq("development")]
        if "segment_type" in subset:
            subset = subset[subset["segment_type"].eq("full")]
        matrix = subset.pivot_table(index="feature", columns="horizon_s", values="spearman", aggfunc="first")
        if matrix.empty:
            raise ValueError("no development IC table")
        fig, ax = plt.subplots(figsize=(8, max(7, .28 * len(matrix))))
        sns.heatmap(matrix, cmap="vlag", center=0, ax=ax); ax.set_title("Feature-to-target rank IC")
        fig.tight_layout(); fig.savefig(paths[4], dpi=dpi); plt.close(fig)

    def fold_ic() -> None:
        subset = fold[(fold.get("task") == "regression") & (fold.get("metric_scope", "fold") == "fold")]
        if subset.empty:
            raise ValueError("no fold metrics")
        fig, ax = plt.subplots(figsize=(11, 6))
        sns.lineplot(subset, x="fold", y="spearman", hue="model", style="horizon_s", markers=True, ax=ax)
        ax.axhline(0, color="black", linewidth=.8); ax.set_title("Rank IC by walk-forward fold")
        fig.tight_layout(); fig.savefig(paths[5], dpi=dpi); plt.close(fig)

    def rolling_ic() -> None:
        subset = predictions[(predictions.get("model") == primary_model) & (predictions.get("horizon_s") == primary_horizon_s)].copy()
        if len(subset) < 30:
            raise ValueError("too few primary predictions")
        subset = subset.sort_values("timestamp")
        window = min(900, max(30, len(subset) // 5))
        roll = subset["prediction"].rolling(window).corr(subset["actual"])
        fig, ax = plt.subplots(figsize=(12, 5)); ax.plot(pd.to_datetime(subset["timestamp"], utc=True), roll)
        ax.axhline(0, color="black", linewidth=.8); ax.set(title=f"Rolling IC ({window} observations)", ylabel="Pearson IC")
        fig.tight_layout(); fig.savefig(paths[6], dpi=dpi); plt.close(fig)

    def prediction_scatter() -> None:
        subset = predictions[(predictions.get("model") == primary_model) & (predictions.get("horizon_s") == primary_horizon_s)].dropna(subset=["prediction", "actual"])
        if subset.empty:
            raise ValueError("no primary predictions")
        if len(subset) > 20_000:
            subset = subset.sample(20_000, random_state=42)
        fig, ax = plt.subplots(figsize=(7, 7)); ax.scatter(subset["prediction"] * 1e4, subset["actual"] * 1e4, s=7, alpha=.25)
        ax.axhline(0, color="grey", linewidth=.7); ax.axvline(0, color="grey", linewidth=.7)
        ax.set(title="Prediction versus realized return", xlabel="prediction (bps)", ylabel="realized (bps)")
        fig.tight_layout(); fig.savefig(paths[7], dpi=dpi); plt.close(fig)

    def quantile_returns() -> None:
        subset = quantiles[(quantiles.get("scope") == "test") & (quantiles.get("model") == primary_model) & (quantiles.get("horizon_s") == primary_horizon_s)]
        if subset.empty:
            raise ValueError("no primary test quantiles")
        fig, ax = plt.subplots(figsize=(10, 5.5)); ax.errorbar(subset["bucket"], subset["mean_return"] * 1e4, yerr=subset.get("std_return", 0) / np.sqrt(subset["n"].clip(lower=1)) * 1e4, marker="o")
        ax.axhline(0, color="black", linewidth=.8); ax.set(title="Mean realized return by frozen signal bucket", xlabel="ascending signal bucket", ylabel="mean return (bps)")
        fig.tight_layout(); fig.savefig(paths[8], dpi=dpi); plt.close(fig)

    def cumulative(column: str, path: Path, title: str, cost: float) -> None:
        subset = trades[(trades.get("model") == primary_model) & (trades.get("horizon_s") == primary_horizon_s) & (trades.get("execution") == "taker") & np.isclose(trades.get("cost_bps", np.nan), cost)].sort_values("exit_time")
        if subset.empty or column not in subset:
            raise ValueError("no matching trades")
        fig, ax = plt.subplots(figsize=(12, 5)); ax.plot(pd.to_datetime(subset["exit_time"], utc=True), subset[column].cumsum() * 1e4)
        ax.axhline(0, color="black", linewidth=.8); ax.set(title=title, ylabel="cumulative log PnL (bps)")
        fig.tight_layout(); fig.savefig(path, dpi=dpi); plt.close(fig)

    def drawdown() -> None:
        subset = trades[(trades.get("model") == primary_model) & (trades.get("horizon_s") == primary_horizon_s) & (trades.get("execution") == "taker") & np.isclose(trades.get("cost_bps", np.nan), confirmatory_cost_bps)].sort_values("exit_time")
        if subset.empty:
            raise ValueError("no primary trades")
        equity = np.exp(subset["net_return"].cumsum()); dd = equity / equity.cummax() - 1
        fig, ax = plt.subplots(figsize=(12, 5)); ax.fill_between(pd.to_datetime(subset["exit_time"], utc=True), dd * 100, 0, alpha=.5)
        ax.set(title="Taker diagnostic drawdown", ylabel="drawdown (%)"); fig.tight_layout(); fig.savefig(paths[11], dpi=dpi); plt.close(fig)

    def cost_plot() -> None:
        subset = costs[(costs.get("stage") == "test") & (costs.get("execution") == "taker")]
        if subset.empty:
            raise ValueError("no cost table")
        fig, ax = plt.subplots(figsize=(11, 6)); sns.lineplot(subset, x="cost_bps", y="net_pnl", hue="model", style="horizon_s", marker="o", ax=ax)
        ax.axhline(0, color="black", linewidth=.8); ax.set(title="Incremental round-trip cost sensitivity", ylabel="net log PnL")
        fig.tight_layout(); fig.savefig(paths[12], dpi=dpi); plt.close(fig)

    def model_ic() -> None:
        subset = fold[(fold.get("metric_scope") == "test") & (fold.get("task") == "regression")]
        if subset.empty:
            raise ValueError("no test metrics")
        fig, ax = plt.subplots(figsize=(11, 6)); sns.barplot(subset, x="model", y="spearman", hue="horizon_s", ax=ax)
        ax.axhline(0, color="black", linewidth=.8); ax.tick_params(axis="x", rotation=35); ax.set_title("Untouched-test rank IC by model")
        fig.tight_layout(); fig.savefig(paths[13], dpi=dpi); plt.close(fig)

    def model_pnl() -> None:
        subset = trading[(trading.get("stage") == "test") & (trading.get("execution") == "taker") & np.isclose(trading.get("cost_bps", np.nan), confirmatory_cost_bps)]
        if subset.empty:
            raise ValueError("no model PnL metrics")
        fig, ax = plt.subplots(figsize=(11, 6)); sns.barplot(subset, x="model", y="net_pnl", hue="horizon_s", ax=ax)
        ax.axhline(0, color="black", linewidth=.8); ax.tick_params(axis="x", rotation=35); ax.set_title("Test taker net PnL by model")
        fig.tight_layout(); fig.savefig(paths[14], dpi=dpi); plt.close(fig)

    def ablation_plot() -> None:
        subset = ablation[ablation.get("metric_scope").isin(["validation_mean", "test"])] if "metric_scope" in ablation else ablation
        if subset.empty:
            raise ValueError("no ablation table")
        fig, ax = plt.subplots(figsize=(11, 6)); sns.barplot(subset, x="feature_group", y="spearman", hue="horizon_s", ax=ax)
        ax.axhline(0, color="black", linewidth=.8); ax.tick_params(axis="x", rotation=25); ax.set_title("Feature-group ablation rank IC")
        fig.tight_layout(); fig.savefig(paths[15], dpi=dpi); plt.close(fig)

    def importance_plot() -> None:
        subset = importance.groupby("feature", as_index=False)["importance_mean"].mean().nlargest(20, "importance_mean")
        if subset.empty:
            raise ValueError("no permutation importance")
        fig, ax = plt.subplots(figsize=(10, 7)); sns.barplot(subset, y="feature", x="importance_mean", ax=ax); ax.set_title("Validation permutation importance")
        fig.tight_layout(); fig.savefig(paths[16], dpi=dpi); plt.close(fig)

    def confusion_plot() -> None:
        subset = classification[(classification.get("record_type") == "confusion") & (classification.get("metric_scope") == "test")]
        if subset.empty:
            raise ValueError("no confusion data")
        models = list(subset[["model", "horizon_s"]].drop_duplicates().itertuples(index=False, name=None))
        models = models[:6]
        fig, axes = plt.subplots(2, 3, figsize=(13, 8)); axes = axes.ravel()
        for ax, key in zip(axes, models):
            part = subset[(subset["model"] == key[0]) & (subset["horizon_s"] == key[1])]
            matrix = part.pivot_table(index="true_label", columns="predicted_label", values="value", fill_value=0).reindex(index=[-1, 0, 1], columns=[-1, 0, 1], fill_value=0)
            sns.heatmap(matrix, annot=True, fmt="g", cmap="Blues", cbar=False, ax=ax); ax.set_title(f"{key[0]} · {key[1]}s")
        for ax in axes[len(models):]: ax.axis("off")
        fig.suptitle("Untouched-test confusion matrices"); fig.tight_layout(); fig.savefig(paths[17], dpi=dpi); plt.close(fig)

    def regime_plot() -> None:
        subset = regimes[(regimes.get("stage") == "test") & (regimes.get("model") == primary_model) & (regimes.get("horizon_s") == primary_horizon_s)]
        if subset.empty:
            raise ValueError("no regime table")
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        sns.barplot(subset, x="regime", y="spearman", hue="regime_type", ax=axes[0]); axes[0].axhline(0, color="black", linewidth=.8); axes[0].set_title("Rank IC by regime")
        sns.barplot(subset, x="regime", y="net_pnl", hue="regime_type", ax=axes[1]); axes[1].axhline(0, color="black", linewidth=.8); axes[1].set_title("Net PnL by regime")
        fig.tight_layout(); fig.savefig(paths[18], dpi=dpi); plt.close(fig)

    def contribution_plot() -> None:
        subset = contribution[(contribution.get("model") == primary_model) & (contribution.get("horizon_s") == primary_horizon_s)]
        if subset.empty:
            raise ValueError("no contribution table")
        fig, axes = plt.subplots(2, 1, figsize=(13, 8))
        for ax, period_type in zip(axes, ["hour", "day"]):
            part = subset[subset["period_type"] == period_type].sort_values("period")
            ax.bar(part["period"].astype(str), part["net_pnl"] * 1e4); ax.axhline(0, color="black", linewidth=.8); ax.set_title(f"Net PnL contribution by {period_type}"); ax.tick_params(axis="x", rotation=45)
        fig.tight_layout(); fig.savefig(paths[19], dpi=dpi); plt.close(fig)

    jobs = [
        ("Target return distributions", target_distributions),
        ("Spread distribution", spread_distribution),
        ("Trade intensity", trade_intensity),
        ("Feature correlation", feature_heatmap),
        ("Feature-target IC", ic_heatmap),
        ("Fold IC stability", fold_ic),
        ("Rolling IC", rolling_ic),
        ("Prediction scatter", prediction_scatter),
        ("Quantile returns", quantile_returns),
        ("Cumulative gross PnL", lambda: cumulative("gross_return", paths[9], "Cumulative taker gross PnL", 0.0)),
        ("Cumulative net PnL", lambda: cumulative("net_return", paths[10], "Cumulative taker net PnL", confirmatory_cost_bps)),
        ("Drawdown", drawdown),
        ("Cost sensitivity", cost_plot),
        ("Model IC comparison", model_ic),
        ("Model PnL comparison", model_pnl),
        ("Feature ablation", ablation_plot),
        ("Permutation importance", importance_plot),
        ("Confusion matrices", confusion_plot),
        ("Regime performance", regime_plot),
        ("Time contribution", contribution_plot),
    ]
    for path, (title, fn) in zip(paths, jobs):
        _plot_guard(path, title, fn)
    return paths


def render_report(run_dir: str | Path) -> Path:
    run = Path(run_dir).resolve()
    config = yaml.safe_load((run / "config.yaml").read_text(encoding="utf-8"))
    quality = _read_json(run / "data_quality.json", {})
    verdict = _read_json(run / "verdict.json", {"verdict": "NO-GO", "reason_codes": ["MISSING_VERDICT"]})
    lock = _read_json(run / "decision_lock.json", {})
    dataset = _read_json(run / "dataset_manifest.json", {})
    metrics = _read_csv(run / "metrics_summary.csv")
    stability = _read_csv(run / "feature_stability.csv")
    trading = _read_csv(run / "trading_metrics.csv")
    ablation = _read_csv(run / "ablation_metrics.csv")
    concentration = _read_csv(run / "concentration_metrics.csv")

    primary_model = lock.get("primary_model_selected", config["models"]["primary_regression_model"])
    horizon = int(lock.get("primary_horizon_s", config["models"]["primary_horizon_s"]))
    cost = float(config["verdict"]["confirmatory_incremental_cost_bps"])
    primary = metrics[
        (metrics.get("metric_scope") == "test")
        & (metrics.get("model") == primary_model)
        & (metrics.get("horizon_s") == horizon)
    ]
    p = primary.iloc[0].to_dict() if not primary.empty else {}
    trade = trading[
        (trading.get("stage") == "test")
        & (trading.get("model") == primary_model)
        & (trading.get("horizon_s") == horizon)
        & (trading.get("execution") == "taker")
        & np.isclose(trading.get("cost_bps", np.nan), cost)
    ]
    t = trade.iloc[0].to_dict() if not trade.empty else {}
    all_test = metrics[metrics.get("metric_scope") == "test"]
    best = all_test.sort_values("spearman", ascending=False).head(1)
    b = best.iloc[0].to_dict() if not best.empty else {}
    micro = ablation[
        (ablation.get("metric_scope") == "test")
        & (ablation.get("horizon_s") == horizon)
    ]
    ablation_text = "not available"
    if not micro.empty and {"feature_group", "spearman"}.issubset(micro):
        pairs = {row.feature_group: row.spearman for row in micro.itertuples()}
        if "all" in pairs and "price_history" in pairs:
            ablation_text = f"all minus price-history rank IC = {_fmt(pairs['all'] - pairs['price_history'])}"

    interval = dataset.get("data_interval", {})
    processed_quality = quality.get("processed", {})
    raw_totals = quality.get("raw", {}).get("totals", {})
    raw_streams = quality.get("raw", {}).get("streams", {})
    duration_h = (processed_quality.get("duration_s") or 0) / 3600
    usable_pct = processed_quality.get("book_usable_pct")
    reason_codes = ", ".join(verdict.get("reason_codes", [])) or "none"
    warnings = quality.get("reasons", [])
    warning_lines = "\n".join(f"- {item}" for item in warnings) or "- None recorded."

    lines = [
        "# Microstructure Alpha Feasibility Probe",
        "",
        "> This is a feasibility diagnostic, not a production strategy or a claim about executable returns.",
        "",
        "## Dataset",
        "",
        f"Session covers **{_fmt(duration_h, 2)} hours** ({interval.get('first_timestamp', 'n/a')} to {interval.get('last_timestamp', 'n/a')}). "
        f"Usable book observations: **{_fmt(usable_pct, 2)}%**. Dataset fingerprint: `{dataset.get('sha256', 'n/a')}`.",
        "",
        "## Data quality",
        "",
        f"Status: **{quality.get('status', 'UNKNOWN')}**. Longest raw gap: {_fmt(max((stream.get('longest_missing_interval_s') or 0 for stream in raw_streams.values()), default=0), 3)} s; "
        f"reconnects: {raw_totals.get('reconnect_count', 'n/a')}; resynchronizations: {raw_totals.get('resynchronization_count', 'n/a')}; malformed messages: {raw_totals.get('malformed_count', 'n/a')}.",
        "",
        warning_lines,
        "",
        "## Predictive signal",
        "",
        f"The predeclared primary pair is **{primary_model} at {horizon}s**: test Pearson IC {_fmt(p.get('pearson'))}, rank IC {_fmt(p.get('spearman'))}, RMSE {_fmt(p.get('rmse'), 7)}, sign accuracy {_fmt(p.get('sign_accuracy'))}, on {_fmt(p.get('n'), 0)} rows. "
        f"The descriptively best test row is {b.get('model', 'n/a')} at {b.get('horizon_s', 'n/a')}s with rank IC {_fmt(b.get('spearman'))} on {_fmt(b.get('n'), 0)} rows; it was not used to redefine the confirmatory pair.",
        "",
        "Multiple comparisons warning: feature, horizon, model, regime and time-slice searches inflate apparent significance. Read `multiple_testing.csv`; raw IID p-values are only approximations and do not drive the verdict.",
        "",
        "## Stability",
        "",
        f"Primary fold sign agreement and IC dispersion are in `fold_metrics.csv` and `bootstrap_confidence_intervals.csv`. "
        f"Feature stability has {len(stability):,} rows; no average is a substitute for the individual folds.",
        "",
        "## Economic relevance",
        "",
        f"Under the non-overlapping taker approximation and **{cost:g} bps additional round-trip fee/slippage**, primary test net log PnL is {_fmt(t.get('net_pnl'), 6)}, mean per trade {_fmt(t.get('mean_return'), 7)}, across {t.get('number_of_trades', 'n/a')} trades. "
        "The taker approximation crosses the recorded bid/ask but still assumes zero decision latency and available size; mid-price diagnostics are optimistic only.",
        "",
        "## Concentration",
        "",
        f"Concentration evidence is saved in `concentration_metrics.csv` ({len(concentration):,} rows) and `contribution_metrics.csv`. A positive total dominated by one hour or a tiny tail is treated as failure, not alpha.",
        "",
        "## Feature evidence",
        "",
        f"Predeclared group comparison: **{ablation_text}**. Inspect `ablation_metrics.csv`, `feature_target_correlations.csv`, and `permutation_importance.csv` before attributing signal to microstructure.",
        "",
        "## Final verdict",
        "",
        f"### {verdict.get('verdict', 'NO-GO')}",
        "",
        verdict.get("summary", "No valid automated verdict was available."),
        "",
        f"Reason codes: `{reason_codes}`.",
        "",
        "The automated decision is a transparent screen. All component flags and values are in `verdict.json`; a successful smoke run can only verify plumbing and must not be interpreted as alpha.",
        "",
        "## Evidence map",
        "",
        "- Start with `data_quality_report.md`, `verdict.json`, `metrics_summary.csv`, `fold_metrics.csv`, and `quantile_analysis.csv`.",
        "- Then inspect `trading_metrics.csv`, `cost_sensitivity.csv`, `concentration_metrics.csv`, and `ablation_metrics.csv`.",
        "- Audit uncertainty/FDR in `bootstrap_confidence_intervals.csv` and `multiple_testing.csv`.",
        "- Exact test predictions and completed diagnostic trades remain in Parquet.",
        "",
        "## Plots",
        "",
    ]
    lines.extend(f"![{name.removesuffix('.png').replace('_', ' ')}](plots/{name})" for name in PLOT_NAMES)
    path = run / "report.md"
    _replace_text(path, "\n".join(lines) + "\n")
    return path
