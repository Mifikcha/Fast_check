"""Leakage-hostile statistical evaluation and immutable run artifacts."""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix, f1_score, matthews_corrcoef, mean_absolute_error,
    mean_squared_error, precision_recall_fscore_support, r2_score,
    roc_auc_score, average_precision_score,
)
from sklearn.preprocessing import label_binarize

from .config import PROJECT_ROOT, dump_config, project_path
from .diagnostics import data_quality_payloads, generate_data_quality_payload
from .io_utils import (
    atomic_write_dataframe, atomic_write_json, atomic_write_text, create_unique_dir,
    dataset_fingerprint, environment_manifest, git_commit, list_parquet_parts,
    session_id as make_run_id, sha256_file,
)
from .models import (
    Candidate, aligned_class_probabilities, build_classifier, build_regressor,
    builtin_importance, classification_candidates, majority_class_prediction,
    naive_regression_prediction, regression_candidates, resolve_feature_groups,
    standardized_coefficients,
)
from .report import generate_plots, render_report
from .splits import make_chronological_split, make_walk_forward_splits
from .statistics import (
    apply_bh_fdr, bootstrap_correlation, descriptive_statistics,
    feature_target_statistics, fit_quantile_edges, fold_ic_stability,
    hac_correlation_inference, hac_mean_inference, moving_block_bootstrap,
    quantile_analysis, safe_correlation,
)
from .trading_diagnostic import (
    assign_regimes, build_non_overlapping_trades, concentration_metrics,
    fit_regime_edges, select_validation_threshold, trading_metrics,
)


LABELS = np.array([-1, 0, 1], dtype=int)


def _finite(value: Any) -> float:
    try:
        number = float(value)
        return number if np.isfinite(number) else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, (pd.Timestamp, pd.Timedelta, Path)):
        return str(value)
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    atomic_write_dataframe(frame, path)


def _read_raw(config: dict[str, Any], kind: str, capture: str) -> pd.DataFrame:
    """Read diagnostics columns only; authoritative JSON remains in raw parts."""
    paths = list_parquet_parts(project_path(config, "raw_dir") / kind / capture)
    if not paths:
        return pd.DataFrame()
    excluded = {"raw_json", "bids_json", "asks_json", "payload_hash"}
    import pyarrow.parquet as pq

    frames = []
    for path in paths:
        columns = [name for name in pq.read_schema(path).names if name not in excluded]
        frames.append(pd.read_parquet(path, columns=columns))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def regression_metrics(actual: Sequence[float], prediction: Sequence[float]) -> dict[str, Any]:
    y = pd.to_numeric(pd.Series(actual), errors="coerce").to_numpy(float)
    p = pd.to_numeric(pd.Series(prediction), errors="coerce").to_numpy(float)
    keep = np.isfinite(y) & np.isfinite(p)
    y, p = y[keep], p[keep]
    if not len(y):
        return {key: float("nan") for key in (
            "pearson", "spearman", "mae", "rmse", "r2", "sign_accuracy",
            "mean_prediction", "std_prediction", "mean_actual", "std_actual",
        )} | {"n": 0}
    return {
        "n": int(len(y)),
        "pearson": safe_correlation(p, y, "pearson"),
        "spearman": safe_correlation(p, y, "spearman"),
        "mae": float(mean_absolute_error(y, p)),
        "rmse": float(math.sqrt(mean_squared_error(y, p))),
        "r2": float(r2_score(y, p)) if len(y) >= 2 and np.ptp(y) > 0 else float("nan"),
        "sign_accuracy": float(np.mean(np.sign(p) == np.sign(y))),
        "mean_prediction": float(np.mean(p)),
        "std_prediction": float(np.std(p, ddof=1)) if len(p) >= 2 else float("nan"),
        "mean_actual": float(np.mean(y)),
        "std_actual": float(np.std(y, ddof=1)) if len(y) >= 2 else float("nan"),
    }


def classification_metrics(
    actual: Sequence[int], prediction: Sequence[int], probabilities: np.ndarray,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    y, p = np.asarray(actual, dtype=int), np.asarray(prediction, dtype=int)
    precision, recall, f1, support = precision_recall_fscore_support(
        y, p, labels=LABELS, zero_division=0
    )
    result: dict[str, Any] = {
        "n": int(len(y)),
        "accuracy": float(accuracy_score(y, p)),
        "balanced_accuracy": float(np.mean(recall[support > 0])),
        "macro_f1": float(f1_score(y, p, labels=LABELS, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y, p, labels=LABELS, average="weighted", zero_division=0)),
        "mcc": float(matthews_corrcoef(y, p)) if np.unique(y).size > 1 else float("nan"),
    }
    names = {-1: "short", 0: "flat", 1: "long"}
    for index, label in enumerate(LABELS):
        result[f"precision_{names[label]}"] = float(precision[index])
        result[f"recall_{names[label]}"] = float(recall[index])
        result[f"f1_{names[label]}"] = float(f1[index])
        result[f"support_{names[label]}"] = int(support[index])
    one_hot = label_binarize(y, classes=LABELS)
    result["brier_score_multiclass"] = float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))
    if set(np.unique(y)) == set(LABELS):
        result["roc_auc_ovr_macro"] = float(roc_auc_score(one_hot, probabilities, average="macro", multi_class="ovr"))
        result["pr_auc_macro"] = float(average_precision_score(one_hot, probabilities, average="macro"))
    else:
        result["roc_auc_ovr_macro"] = float("nan")
        result["pr_auc_macro"] = float("nan")

    matrix = confusion_matrix(y, p, labels=LABELS)
    confusion = pd.DataFrame(
        [
            {"true_label": int(true), "predicted_label": int(predicted), "value": int(matrix[i, j])}
            for i, true in enumerate(LABELS) for j, predicted in enumerate(LABELS)
        ]
    )
    calibration_rows: list[dict[str, Any]] = []
    for j, label in enumerate(LABELS):
        bins = pd.cut(probabilities[:, j], np.linspace(0, 1, 11), include_lowest=True, duplicates="drop")
        for bucket in bins.categories:
            mask = bins == bucket
            if mask.any():
                calibration_rows.append({
                    "class_label": int(label), "probability_bin": str(bucket),
                    "n": int(mask.sum()), "mean_predicted_probability": float(probabilities[mask, j].mean()),
                    "observed_frequency": float(np.mean(y[mask] == label)),
                })
    return result, confusion, pd.DataFrame(calibration_rows)


def _valid_mask(frame: pd.DataFrame, horizon: int, base: pd.Series | np.ndarray, require_path: bool) -> pd.Series:
    mask = pd.Series(base, index=frame.index).fillna(False).astype(bool)
    mask &= frame["usable"].fillna(False).astype(bool)
    mask &= frame[f"target_valid_{horizon}s"].fillna(False).astype(bool)
    if require_path:
        mask &= frame[f"target_path_valid_{horizon}s"].fillna(False).astype(bool)
    mask &= np.isfinite(pd.to_numeric(frame[f"target_return_{horizon}s"], errors="coerce"))
    return mask


def _candidate_predictions(
    candidate: Candidate, train: pd.DataFrame, evaluation: pd.DataFrame,
    features: list[str], target: str, *, seed: int, n_jobs: int,
) -> tuple[np.ndarray, Any | None]:
    if candidate.family == "naive":
        return naive_regression_prediction(
            candidate, train[target].to_numpy(float), evaluation
        ), None
    model = build_regressor(candidate, seed=seed, n_jobs=n_jobs)
    model.fit(train[features], train[target])
    return model.predict(evaluation[features]), model


def _choose_by_family(fold_metrics: pd.DataFrame, candidates: list[Candidate], horizon: int) -> dict[str, Candidate]:
    lookup = {candidate.name: candidate for candidate in candidates}
    part = fold_metrics[
        (fold_metrics["task"] == "regression")
        & (fold_metrics["metric_scope"] == "fold")
        & (fold_metrics["horizon_s"] == horizon)
    ]
    means = part.groupby("model", as_index=False).agg(spearman=("spearman", "mean"), rmse=("rmse", "mean"))
    selected: dict[str, Candidate] = {}
    for family in ("naive", "linear", "ridge", "lightgbm"):
        names = [candidate.name for candidate in candidates if candidate.family == family]
        eligible = means[means["model"].isin(names)].sort_values(
            ["spearman", "rmse", "model"], ascending=[False, True, True], kind="stable"
        )
        if not eligible.empty:
            selected[family] = lookup[str(eligible.iloc[0]["model"])]
    return selected


def _walk_folds(development: pd.DataFrame, config: dict[str, Any], max_horizon: int) -> tuple[list[dict[str, object]], int]:
    configured = int(config["splits"]["walk_forward_folds"])
    # Each assessment block needs both a leading embargo and a purged target tail.
    possible = max(1, len(development) // (2 * max_horizon + 2) - 1)
    used = min(configured, possible)
    return make_walk_forward_splits(
        development, n_splits=used, max_horizon_s=max_horizon,
        embargo_s=int(config["splits"]["embargo_s"]),
    ), used


def _fold_regression(
    development: pd.DataFrame, folds: list[dict[str, object]], candidates: list[Candidate],
    features: list[str], horizons: list[int], config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    predictions: list[pd.DataFrame] = []
    require_path = bool(config["targets"]["require_valid_path"])
    seed, n_jobs = int(config["project"]["seed"]), int(config["models"]["lightgbm_n_jobs"])
    minimum_train = int(config["splits"].get("minimum_fold_train_rows", 1))
    for fold in folds:
        fold_no = int(fold["fold"])
        for horizon in horizons:
            target = f"target_return_{horizon}s"
            train_mask = _valid_mask(development, horizon, fold["train"], require_path)
            validation_mask = _valid_mask(development, horizon, fold["validation"], require_path)
            train, validation = development.loc[train_mask], development.loc[validation_mask]
            for candidate in candidates:
                base = {"task": "regression", "metric_scope": "fold", "fold": fold_no,
                        "model": candidate.name, "family": candidate.family, "horizon_s": horizon}
                if len(train) < minimum_train or validation.empty:
                    rows.append(base | regression_metrics([], []) | {"error": "insufficient_valid_train_or_validation"})
                    continue
                try:
                    prediction, _ = _candidate_predictions(
                        candidate, train, validation, features, target,
                        seed=seed + 1_000 * fold_no + horizon, n_jobs=n_jobs,
                    )
                    rows.append(base | regression_metrics(validation[target], prediction) | {"error": None})
                    prediction_frame = validation[[
                        "timestamp", "mid", "best_bid", "best_ask", "book_valid",
                        f"target_end_time_{horizon}s", f"future_mid_{horizon}s",
                        f"future_bid_{horizon}s", f"future_ask_{horizon}s",
                        f"target_valid_{horizon}s", f"target_path_valid_{horizon}s",
                    ]].copy()
                    prediction_frame["model"] = candidate.name
                    prediction_frame["horizon_s"] = horizon
                    prediction_frame["fold"] = fold_no
                    prediction_frame["prediction"] = prediction
                    prediction_frame["actual"] = validation[target].to_numpy(float)
                    prediction_frame["actual_class"] = validation[f"target_class_{horizon}s"].to_numpy()
                    predictions.append(prediction_frame)
                except Exception as exc:
                    rows.append(base | regression_metrics([], []) | {"error": f"{type(exc).__name__}: {exc}"})
    return pd.DataFrame(rows), pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()


def _fold_classification(
    development: pd.DataFrame, folds: list[dict[str, object]], features: list[str],
    horizons: list[int], config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[int, Candidate]]:
    candidates = classification_candidates(config)
    rows: list[dict[str, Any]] = []
    require_path = bool(config["targets"]["require_valid_path"])
    seed = int(config["project"]["seed"])
    minimum_train = int(config["splits"].get("minimum_fold_train_rows", 1))
    for fold in folds:
        for horizon in horizons:
            train_mask = _valid_mask(development, horizon, fold["train"], require_path)
            valid_mask = _valid_mask(development, horizon, fold["validation"], require_path)
            train, valid = development.loc[train_mask], development.loc[valid_mask]
            y_train = train[f"target_class_{horizon}s"].astype(int).to_numpy()
            y_valid = valid[f"target_class_{horizon}s"].astype(int).to_numpy()
            for candidate in candidates:
                base = {"record_type": "summary", "metric_scope": "fold", "fold": int(fold["fold"]),
                        "model": candidate.name, "family": candidate.family, "horizon_s": horizon}
                if len(y_train) < minimum_train or not len(y_valid):
                    rows.append(base | {"n": 0, "macro_f1": np.nan, "error": "insufficient_valid_train_or_validation"})
                    continue
                try:
                    if candidate.family == "naive":
                        pred, prob = majority_class_prediction(y_train, len(valid))
                    else:
                        model = build_classifier(candidate, seed=seed + horizon + int(fold["fold"]), n_jobs=1)
                        model.fit(train[features], y_train)
                        pred, prob = model.predict(valid[features]), aligned_class_probabilities(model, valid[features])
                    metrics, _, _ = classification_metrics(y_valid, pred, prob)
                    rows.append(base | metrics | {"error": None})
                except Exception as exc:
                    rows.append(base | {"n": 0, "macro_f1": np.nan, "error": f"{type(exc).__name__}: {exc}"})
    table = pd.DataFrame(rows)
    selected: dict[int, Candidate] = {}
    lookup = {candidate.name: candidate for candidate in candidates}
    for horizon in horizons:
        part = table[(table["horizon_s"] == horizon) & table["model"].str.startswith("logistic_")]
        scores = part.groupby("model", as_index=False)["macro_f1"].mean().sort_values(["macro_f1", "model"], ascending=[False, True])
        if not scores.empty:
            selected[horizon] = lookup[str(scores.iloc[0]["model"])]
    return table, selected


def _univariate_tables(
    research: pd.DataFrame, development: pd.DataFrame, outer: dict[str, object],
    folds: list[dict[str, object]], features: list[str], horizons: list[int], config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    stats_table = descriptive_statistics(
        development, features,
        autocorrelation_lags=[int(x) for x in config["statistics"]["autocorrelation_lags_s"]],
    )
    targets = [f"target_return_{h}s" for h in horizons]
    correlation_frames: list[pd.DataFrame] = []

    def add(part: pd.DataFrame, scope: str, segment_type: str, segment: Any) -> None:
        if len(part) < int(config["statistics"]["minimum_group_n"]):
            return
        table = feature_target_statistics(
            part, features, targets,
            sampling_interval_s=int(config["sampling"]["interval_s"]),
            max_mi_rows=int(config["statistics"]["max_mutual_information_rows"]),
            seed=int(config["project"]["seed"]), stage=scope,
        )
        table.insert(0, "scope", scope)
        table.insert(1, "segment_type", segment_type)
        table.insert(2, "segment", str(segment))
        correlation_frames.append(table)

    add(development, "development", "full", "all")
    add(research.loc[outer["test"]], "test", "full", "all")
    for fold in folds:
        add(development.loc[fold["validation"]], "development", "fold", int(fold["fold"]))
    for day, part in development.groupby(development["timestamp"].dt.floor("D")):
        add(part, "development", "day", day.isoformat())
    for hour, part in development.groupby(development["timestamp"].dt.hour):
        add(part, "development", "hour_of_day", int(hour))
    correlations = pd.concat(correlation_frames, ignore_index=True) if correlation_frames else pd.DataFrame()

    multiple_frames: list[pd.DataFrame] = []
    if not correlations.empty:
        full = correlations[correlations["segment_type"].eq("full")]
        for association, p_column in (("pearson", "pearson_p_hac"), ("spearman", "spearman_p_hac")):
            part = full[["scope", "feature", "target", "horizon_s", p_column]].rename(columns={p_column: "p_value"}).copy()
            part["association_type"] = association
            part = apply_bh_fdr(
                part, p_column="p_value", alpha=float(config["statistics"]["fdr_alpha"]),
                family_columns=["scope", "association_type"],
            )
            multiple_frames.append(part)
    multiple = pd.concat(multiple_frames, ignore_index=True) if multiple_frames else pd.DataFrame(
        columns=["scope", "feature", "target", "horizon_s", "p_value", "association_type", "q_value", "reject_fdr"]
    )
    fold_corr = correlations[(correlations.get("scope") == "development") & (correlations.get("segment_type") == "fold")].copy()
    if not fold_corr.empty:
        fold_corr["fold"] = pd.to_numeric(fold_corr["segment"], errors="coerce")
        stability = fold_ic_stability(fold_corr, group_columns=["feature", "target"], ic_column="spearman")
    else:
        stability = pd.DataFrame(columns=["feature", "target", "fold_count", "mean_ic"])
    return stats_table, correlations, multiple, stability


def _summary_from_folds(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    numeric = [
        "n", "pearson", "spearman", "mae", "rmse", "r2", "sign_accuracy",
        "mean_prediction", "std_prediction", "mean_actual", "std_actual",
    ]
    rows: list[dict[str, Any]] = []
    for keys, part in fold_metrics[fold_metrics["metric_scope"].eq("fold")].groupby(["model", "family", "horizon_s"], dropna=False):
        base = {"model": keys[0], "family": keys[1], "horizon_s": keys[2], "metric_scope": "validation_mean"}
        rows.append(base | {name: pd.to_numeric(part[name], errors="coerce").mean() for name in numeric} | {"fold_count": len(part)})
        rows.append(base | {"metric_scope": "validation_std"} | {name: pd.to_numeric(part[name], errors="coerce").std() for name in numeric} | {"fold_count": len(part)})
    return pd.DataFrame(rows)


def _outer_development_mask(research: pd.DataFrame, boundary: pd.Timestamp, max_horizon: int) -> pd.Series:
    return (research["timestamp"] < boundary) & (research["timestamp"] + pd.Timedelta(seconds=max_horizon) < boundary)


def _select_thresholds_and_quantiles(
    oof: pd.DataFrame, selected: dict[int, dict[str, Candidate]], config: dict[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame]:
    thresholds: dict[str, Any] = {}
    selection_tables: list[pd.DataFrame] = []
    for horizon, families in selected.items():
        seen: set[str] = set()
        for candidate in families.values():
            if candidate.name in seen:
                continue
            seen.add(candidate.name)
            part = oof[(oof["model"] == candidate.name) & (oof["horizon_s"] == horizon)].copy()
            if part.empty:
                continue
            key = f"{candidate.name}|{horizon}"
            selection = select_validation_threshold(
                part, prediction_column="prediction", horizon_s=horizon,
                threshold_grid_bps=config["trading"]["threshold_grid_bps"],
                incremental_cost_bps=float(config["trading"]["selection_incremental_cost_bps"]),
                minimum_trades=int(config["trading"]["minimum_validation_trades"]),
                require_path_valid=bool(config["targets"]["require_valid_path"]),
            )
            threshold = selection["selected_threshold"]
            fallback = not np.isfinite(threshold)
            if fallback:
                threshold = 0.0
            try:
                edges = fit_quantile_edges(part["prediction"], int(config["quantiles"]["bins"]))
            except ValueError:
                edges = np.array([-np.inf, np.inf])
            thresholds[key] = {
                "threshold": float(threshold),
                "threshold_bps": float(threshold * 10_000),
                "selection_reason": selection["reason"],
                "descriptive_zero_fallback": fallback,
                "quantile_edges": edges.tolist(),
            }
            table = selection["selection_table"].copy()
            table["model"], table["horizon_s"], table["stage"] = candidate.name, horizon, "validation"
            selection_tables.append(table)
    return thresholds, pd.concat(selection_tables, ignore_index=True) if selection_tables else pd.DataFrame()


def _final_regression(
    research: pd.DataFrame, outer: dict[str, object], selected: dict[int, dict[str, Candidate]],
    features: list[str], config: dict[str, Any], max_horizon: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[tuple[str, int], Any]]:
    test_rows: list[dict[str, Any]] = []
    predictions: list[pd.DataFrame] = []
    fitted: dict[tuple[str, int], Any] = {}
    boundary = pd.Timestamp(outer["validation_test_boundary"])
    development_base = _outer_development_mask(research, boundary, max_horizon)
    require_path = bool(config["targets"]["require_valid_path"])
    seed, n_jobs = int(config["project"]["seed"]), int(config["models"]["lightgbm_n_jobs"])
    for horizon, families in selected.items():
        target = f"target_return_{horizon}s"
        train = research.loc[_valid_mask(research, horizon, development_base, require_path)]
        test = research.loc[_valid_mask(research, horizon, outer["test"], require_path)]
        seen: set[str] = set()
        for candidate in families.values():
            if candidate.name in seen:
                continue
            seen.add(candidate.name)
            base = {"task": "regression", "metric_scope": "test", "fold": "test",
                    "model": candidate.name, "family": candidate.family, "horizon_s": horizon}
            if train.empty or test.empty:
                test_rows.append(base | regression_metrics([], []) | {"error": "empty_valid_train_or_test"})
                continue
            try:
                prediction, model = _candidate_predictions(
                    candidate, train, test, features, target,
                    seed=seed + 10_000 + horizon, n_jobs=n_jobs,
                )
                if model is not None:
                    fitted[(candidate.name, horizon)] = model
                test_rows.append(base | regression_metrics(test[target], prediction) | {"error": None})
                result = test[[
                    "timestamp", "mid", "best_bid", "best_ask", "book_valid",
                    "spread_bps", config["regimes"]["volatility_feature"],
                    f"target_end_time_{horizon}s", f"future_mid_{horizon}s",
                    f"future_bid_{horizon}s", f"future_ask_{horizon}s",
                    f"target_valid_{horizon}s", f"target_path_valid_{horizon}s",
                ]].copy()
                result["model"], result["family"], result["horizon_s"] = candidate.name, candidate.family, horizon
                result["prediction"], result["actual"] = prediction, test[target].to_numpy(float)
                result["actual_class"] = test[f"target_class_{horizon}s"].astype(int).to_numpy()
                predictions.append(result)
            except Exception as exc:
                test_rows.append(base | regression_metrics([], []) | {"error": f"{type(exc).__name__}: {exc}"})
    return pd.DataFrame(test_rows), pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame(), fitted


def _final_classification(
    research: pd.DataFrame, outer: dict[str, object], selected: dict[int, Candidate],
    features: list[str], horizons: list[int], config: dict[str, Any], max_horizon: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    boundary = pd.Timestamp(outer["validation_test_boundary"])
    development_base = _outer_development_mask(research, boundary, max_horizon)
    require_path = bool(config["targets"]["require_valid_path"])
    for horizon in horizons:
        train = research.loc[_valid_mask(research, horizon, development_base, require_path)]
        test = research.loc[_valid_mask(research, horizon, outer["test"], require_path)]
        if train.empty or test.empty:
            continue
        candidates = [Candidate("majority_class", "naive", {})]
        if horizon in selected:
            candidates.append(selected[horizon])
        y_train = train[f"target_class_{horizon}s"].astype(int).to_numpy()
        y_test = test[f"target_class_{horizon}s"].astype(int).to_numpy()
        for candidate in candidates:
            base = {"metric_scope": "test", "model": candidate.name, "family": candidate.family, "horizon_s": horizon}
            try:
                if candidate.family == "naive":
                    pred, prob = majority_class_prediction(y_train, len(test))
                else:
                    model = build_classifier(candidate, seed=int(config["project"]["seed"]) + 20_000 + horizon, n_jobs=1)
                    model.fit(train[features], y_train)
                    pred, prob = model.predict(test[features]), aligned_class_probabilities(model, test[features])
                metrics, confusion, calibration = classification_metrics(y_test, pred, prob)
                rows.append(base | {"record_type": "summary"} | metrics)
                rows.extend((base | {"record_type": "confusion"} | record) for record in confusion.to_dict("records"))
                rows.extend((base | {"record_type": "calibration"} | record) for record in calibration.to_dict("records"))
            except Exception as exc:
                rows.append(base | {"record_type": "summary", "n": 0, "error": f"{type(exc).__name__}: {exc}"})
    return pd.DataFrame(rows)


def _quantile_tables(
    oof: pd.DataFrame, test_predictions: pd.DataFrame, thresholds: dict[str, Any],
    research: pd.DataFrame, outer: dict[str, object], features: list[str], horizons: list[int],
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[pd.DataFrame] = []
    metadata: dict[str, Any] = {}
    for key, frozen in thresholds.items():
        model, horizon_text = key.rsplit("|", 1)
        horizon = int(horizon_text)
        edges = np.asarray(frozen["quantile_edges"], dtype=float)
        for scope, source in (
            ("validation", oof[(oof["model"] == model) & (oof["horizon_s"] == horizon)]),
            ("test", test_predictions[(test_predictions["model"] == model) & (test_predictions["horizon_s"] == horizon)]),
        ):
            if source.empty:
                continue
            table, info = quantile_analysis(
                source["prediction"], source["actual"], edges=edges,
                target_class=source["actual_class"], minimum_bucket_n=int(config["quantiles"]["minimum_bucket_n"]),
            )
            table.insert(0, "scope", scope); table.insert(1, "signal_type", "model")
            table["model"], table["feature"], table["horizon_s"] = model, None, horizon
            for name, value in info.items():
                if np.isscalar(value) and name != "finite_cutpoints":
                    table[name] = value
            rows.append(table)
            metadata[f"model|{model}|{horizon}|{scope}"] = info

    important = [name for name in config["features"]["important_features"] if name in features]
    validation_mask = pd.Series(outer["validation"], index=research.index)
    test_mask = pd.Series(outer["test"], index=research.index)
    for feature in important:
        for horizon in horizons:
            valid = _valid_mask(research, horizon, validation_mask, bool(config["targets"]["require_valid_path"]))
            source = research.loc[valid]
            if source[feature].notna().sum() < 2:
                continue
            correlation = safe_correlation(source[feature], source[f"target_return_{horizon}s"], "spearman")
            orientation = float(np.sign(correlation)) if np.isfinite(correlation) and correlation != 0 else 1.0
            signal = source[feature] * orientation
            try:
                edges = fit_quantile_edges(signal, int(config["quantiles"]["bins"]))
            except ValueError:
                continue
            for scope, mask in (("validation", valid), ("test", _valid_mask(research, horizon, test_mask, bool(config["targets"]["require_valid_path"])))):
                part = research.loc[mask]
                table, info = quantile_analysis(
                    part[feature] * orientation, part[f"target_return_{horizon}s"], edges=edges,
                    target_class=part[f"target_class_{horizon}s"], minimum_bucket_n=int(config["quantiles"]["minimum_bucket_n"]),
                )
                table.insert(0, "scope", scope); table.insert(1, "signal_type", "feature")
                table["model"], table["feature"], table["horizon_s"], table["orientation"] = None, feature, horizon, orientation
                for name, value in info.items():
                    if np.isscalar(value) and name != "finite_cutpoints":
                        table[name] = value
                rows.append(table)
                metadata[f"feature|{feature}|{horizon}|{scope}"] = info | {"orientation": orientation, "edges": edges.tolist()}
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(), metadata


def _trading_tables(
    test_predictions: pd.DataFrame, thresholds: dict[str, Any], config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metrics_rows: list[dict[str, Any]] = []
    cost_rows: list[dict[str, Any]] = []
    trade_frames: list[pd.DataFrame] = []
    concentration_rows: list[dict[str, Any]] = []
    contribution: dict[tuple[str, int, str, pd.Timestamp], dict[str, Any]] = {}
    costs = [float(value) for value in config["trading"]["incremental_round_trip_cost_bps"]]
    confirmatory = float(config["verdict"]["confirmatory_incremental_cost_bps"])
    for key, frozen in thresholds.items():
        model, horizon_text = key.rsplit("|", 1)
        horizon, threshold = int(horizon_text), float(frozen["threshold"])
        part = test_predictions[(test_predictions["model"] == model) & (test_predictions["horizon_s"] == horizon)]
        if part.empty:
            continue
        prediction = pd.to_numeric(part["prediction"], errors="coerce")
        actual = pd.to_numeric(part["actual"], errors="coerce")
        prediction_rank = prediction.rank(method="average")
        actual_rank = actual.rank(method="average")
        rank_product = (prediction_rank - prediction_rank.mean()) * (actual_rank - actual_rank.mean())
        rank_total = float(rank_product.sum())
        for period_type, frequency in (("hour", "h"), ("day", "D")):
            periods = pd.to_datetime(part["timestamp"], utc=True).dt.floor(frequency)
            for period, positions in part.groupby(periods).groups.items():
                group = part.loc[positions]
                product = float(rank_product.loc[positions].sum())
                contribution[(model, horizon, period_type, period)] = {
                    "model": model, "horizon_s": horizon, "period_type": period_type, "period": period,
                    "n": len(group), "pearson": safe_correlation(group["prediction"], group["actual"], "pearson"),
                    "spearman": safe_correlation(group["prediction"], group["actual"], "spearman"),
                    "rank_ic_numerator": product,
                    "rank_ic_numerator_share": product / rank_total if rank_total else np.nan,
                    "net_pnl": 0.0, "number_of_trades": 0,
                }
        for cost in costs:
            trades = build_non_overlapping_trades(
                part, prediction_column="prediction", horizon_s=horizon, threshold=threshold,
                incremental_cost_bps=cost, require_path_valid=bool(config["targets"]["require_valid_path"]),
            )
            for execution in ("mid", "taker"):
                metric = trading_metrics(trades, execution=execution)
                aliases = {"net_pnl": metric["net_pnl_log"], "gross_pnl": metric["gross_pnl_log"],
                           "mean_return": metric["average_net_return"], "cost_bps": cost}
                record = {"stage": "test", "model": model, "horizon_s": horizon,
                          "threshold": threshold, "threshold_bps": threshold * 10_000} | metric | aliases
                metrics_rows.append(record)
                cost_rows.append(record.copy())
                long = trades.copy()
                long["model"], long["horizon_s"], long["execution"], long["cost_bps"] = model, horizon, execution, cost
                long["gross_return"] = long[f"gross_{execution}_log"]
                long["net_return"] = long[f"net_{execution}_log"]
                if not long.empty:
                    trade_frames.append(long)
            if math.isclose(cost, confirmatory):
                concentration_rows.append({"model": model, "horizon_s": horizon, "cost_bps": cost} | concentration_metrics(trades))
                priced = trades[np.isfinite(pd.to_numeric(trades["net_taker_log"], errors="coerce"))].copy()
                if not priced.empty:
                    priced["exit_time"] = pd.to_datetime(priced["exit_time"], utc=True)
                    for period_type, frequency in (("hour", "h"), ("day", "D")):
                        grouped = priced.groupby(priced["exit_time"].dt.floor(frequency))["net_taker_log"].agg(["sum", "count"])
                        for period, row in grouped.iterrows():
                            key = (model, horizon, period_type, period)
                            contribution.setdefault(key, {"model": model, "horizon_s": horizon, "period_type": period_type,
                                                          "period": period, "n": 0, "pearson": np.nan, "spearman": np.nan,
                                                          "rank_ic_numerator": np.nan, "rank_ic_numerator_share": np.nan})
                            contribution[key].update(net_pnl=float(row["sum"]), number_of_trades=int(row["count"]))
    return (
        pd.DataFrame(metrics_rows), pd.DataFrame(cost_rows),
        pd.concat(trade_frames, ignore_index=True) if trade_frames else pd.DataFrame(),
        pd.DataFrame(concentration_rows), pd.DataFrame(contribution.values()),
    )


def _regime_table(
    predictions: pd.DataFrame, trades: pd.DataFrame, development: pd.DataFrame,
    config: dict[str, Any], thresholds: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    vol_name = config["regimes"]["volatility_feature"]
    frozen: dict[str, Any] = {}
    regimes: list[dict[str, Any]] = []
    for regime_type, feature, quantiles in (
        ("volatility", vol_name, config["regimes"]["quantiles"]),
        ("spread", "spread_bps", config["regimes"]["spread_quantiles"]),
    ):
        try:
            edges = fit_regime_edges(development[feature], quantiles)
        except ValueError:
            continue
        labels = ["low", "medium", "high"][: len(edges) - 1]
        frozen[regime_type] = edges.tolist()
        for (model, horizon), part in predictions.groupby(["model", "horizon_s"]):
            part = part.copy()
            part["regime"] = assign_regimes(part[feature], edges, labels)
            threshold = thresholds.get(f"{model}|{int(horizon)}", {}).get("threshold", 0.0)
            for regime, group in part.groupby("regime", observed=False):
                relevant = trades[
                    (trades.get("model") == model) & (trades.get("horizon_s") == horizon)
                    & (trades.get("execution") == "taker")
                    & np.isclose(trades.get("cost_bps", np.nan), float(config["verdict"]["confirmatory_incremental_cost_bps"]))
                ].copy()
                if not relevant.empty:
                    entry_regime = pd.Series(group["timestamp"].to_numpy(), index=group["timestamp"]).index
                    relevant = relevant[pd.to_datetime(relevant["entry_time"], utc=True).isin(entry_regime)]
                regimes.append({
                    "stage": "test", "model": model, "horizon_s": int(horizon), "regime_type": regime_type,
                    "regime": str(regime), "n": len(group),
                    "spearman": safe_correlation(group["prediction"], group["actual"], "spearman"),
                    "sign_accuracy": float(np.mean(np.sign(group["prediction"]) == np.sign(group["actual"]))) if len(group) else np.nan,
                    "number_of_trades": int(len(relevant)),
                    "net_pnl": float(pd.to_numeric(relevant.get("net_return"), errors="coerce").sum()) if "net_return" in relevant else float(pd.to_numeric(relevant.get("net_taker_log"), errors="coerce").sum()),
                    "threshold": threshold,
                })
    return pd.DataFrame(regimes), frozen


def _importance_and_ablation(
    research: pd.DataFrame, development: pd.DataFrame, outer: dict[str, object],
    selected: dict[int, dict[str, Candidate]], fitted: dict[tuple[str, int], Any],
    groups: dict[str, list[str]], horizons: list[int], config: dict[str, Any], max_horizon: int,
    folds: list[dict[str, object]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    importance_rows: list[dict[str, Any]] = []
    permutation_rows: list[dict[str, Any]] = []
    ablation_rows: list[dict[str, Any]] = []
    require_path = bool(config["targets"]["require_valid_path"])
    boundary = pd.Timestamp(outer["validation_test_boundary"])
    dev_base = _outer_development_mask(research, boundary, max_horizon)
    seed = int(config["project"]["seed"])
    for horizon in horizons:
        candidate = selected[horizon].get("ridge") or selected[horizon].get("linear")
        tree = selected[horizon].get("lightgbm")
        for chosen in [candidate, tree]:
            if chosen is None:
                continue
            model = fitted.get((chosen.name, horizon))
            if model is None:
                continue
            if chosen.family in {"linear", "ridge"}:
                table = standardized_coefficients(model, groups["all"]).rename(columns={"coefficient": "importance"})
                table["importance_type"] = "standardized_coefficient"
            else:
                table = builtin_importance(model, groups["all"])
                table["importance_type"] = "lightgbm_gain"
            table["model"], table["horizon_s"] = chosen.name, horizon
            importance_rows.extend(table.to_dict("records"))

        if tree is not None:
            train = research.loc[_valid_mask(research, horizon, outer["train"], require_path)]
            validation = research.loc[_valid_mask(research, horizon, outer["validation"], require_path)]
            if len(train) and len(validation):
                try:
                    model = build_regressor(tree, seed=seed + 30_000 + horizon, n_jobs=int(config["models"]["lightgbm_n_jobs"]))
                    model.fit(train[groups["all"]], train[f"target_return_{horizon}s"])
                    result = permutation_importance(
                        model, validation[groups["all"]], validation[f"target_return_{horizon}s"],
                        scoring="neg_root_mean_squared_error", n_repeats=int(config["models"]["permutation_repeats"]),
                        random_state=seed, n_jobs=1,
                    )
                    for name, mean, std in zip(groups["all"], result.importances_mean, result.importances_std):
                        permutation_rows.append({"model": tree.name, "horizon_s": horizon, "feature": name,
                                                 "importance_mean": float(mean), "importance_std": float(std), "stage": "validation"})
                except Exception as exc:
                    permutation_rows.append({"model": tree.name, "horizon_s": horizon, "feature": None,
                                             "importance_mean": np.nan, "importance_std": np.nan,
                                             "stage": "validation", "error": f"{type(exc).__name__}: {exc}"})

        chosen = tree or candidate
        if chosen is None:
            continue
        for group_name, feature_names in groups.items():
            if not feature_names:
                continue
            train = research.loc[_valid_mask(research, horizon, dev_base, require_path)]
            test = research.loc[_valid_mask(research, horizon, outer["test"], require_path)]
            try:
                fold_results: list[dict[str, Any]] = []
                for fold in folds:
                    fold_train = development.loc[_valid_mask(development, horizon, fold["train"], require_path)]
                    fold_validation = development.loc[_valid_mask(development, horizon, fold["validation"], require_path)]
                    if fold_train.empty or fold_validation.empty:
                        continue
                    fold_prediction, _ = _candidate_predictions(
                        chosen, fold_train, fold_validation, feature_names, f"target_return_{horizon}s",
                        seed=seed + 35_000 + int(fold["fold"]) + horizon,
                        n_jobs=int(config["models"]["lightgbm_n_jobs"]),
                    )
                    fold_metric = regression_metrics(fold_validation[f"target_return_{horizon}s"], fold_prediction)
                    fold_results.append(fold_metric)
                    ablation_rows.append({"metric_scope": "validation_fold", "fold": int(fold["fold"]),
                                          "model": chosen.name, "horizon_s": horizon,
                                          "feature_group": group_name, **fold_metric})
                if fold_results:
                    for scope, reducer in (("validation_mean", np.nanmean), ("validation_std", np.nanstd)):
                        ablation_rows.append({"metric_scope": scope, "fold": None, "model": chosen.name,
                                              "horizon_s": horizon, "feature_group": group_name,
                                              **{name: float(reducer([row.get(name, np.nan) for row in fold_results]))
                                                 for name in regression_metrics([], [])}})
                prediction, _ = _candidate_predictions(
                    chosen, train, test, feature_names, f"target_return_{horizon}s",
                    seed=seed + 40_000 + horizon, n_jobs=int(config["models"]["lightgbm_n_jobs"]),
                )
                ablation_rows.append({"metric_scope": "test", "model": chosen.name, "horizon_s": horizon,
                                      "feature_group": group_name, **regression_metrics(test[f"target_return_{horizon}s"], prediction)})
            except Exception as exc:
                ablation_rows.append({"metric_scope": "test", "model": chosen.name, "horizon_s": horizon,
                                      "feature_group": group_name, "n": 0, "spearman": np.nan, "error": str(exc)})
    return pd.DataFrame(importance_rows), pd.DataFrame(permutation_rows), pd.DataFrame(ablation_rows)


def _uncertainty_and_robustness(
    predictions: pd.DataFrame, trades: pd.DataFrame, development: pd.DataFrame,
    primary_model: str, primary_horizon: int, config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    robustness: list[dict[str, Any]] = []
    n_resamples = int(config["bootstrap"]["n_resamples"])
    confidence = float(config["bootstrap"]["confidence"])
    seed = int(config["project"]["seed"])
    for (model, horizon), part in predictions.groupby(["model", "horizon_s"]):
        horizon = int(horizon)
        for method in ("pearson", "spearman"):
            result = bootstrap_correlation(
                part["prediction"], part["actual"], method=method, block_length=max(1, horizon),
                n_resamples=n_resamples, confidence=confidence, seed=seed + horizon,
            )
            hac = hac_correlation_inference(part["prediction"], part["actual"], max_lag=max(0, horizon - 1), method=method)
            rows.append({"quantity": f"{method}_ic", "model": model, "horizon_s": horizon, "stage": "test", **result,
                         "hac_se": hac["se_hac"], "hac_p_value": hac["p_value_hac"]})

    confirm = float(config["verdict"]["confirmatory_incremental_cost_bps"])
    primary_trades = trades[
        (trades.get("model") == primary_model) & (trades.get("horizon_s") == primary_horizon)
        & (trades.get("execution") == "taker") & np.isclose(trades.get("cost_bps", np.nan), confirm)
    ].copy()
    if not primary_trades.empty:
        start = pd.to_datetime(primary_trades["entry_time"], utc=True).min().floor("s")
        end = pd.to_datetime(primary_trades["exit_time"], utc=True).max().ceil("s")
        calendar = pd.Series(0.0, index=pd.date_range(start, end, freq="1s", tz="UTC"))
        completed = primary_trades[np.isfinite(pd.to_numeric(primary_trades["net_return"], errors="coerce"))]
        for exit_time, value in zip(pd.to_datetime(completed["exit_time"], utc=True), completed["net_return"]):
            calendar.loc[exit_time.floor("s")] += float(value)
        values = calendar.to_numpy(float)
        result = moving_block_bootstrap(
            values, statistic=lambda x: float(np.mean(x)), block_length=max(1, primary_horizon),
            n_resamples=n_resamples, confidence=confidence, seed=seed + 99,
        )
        hac = hac_mean_inference(values, max_lag=min(max(0, primary_horizon - 1), max(0, len(values) - 3)))
        rows.append({"quantity": "mean_taker_trade_return", "model": primary_model, "horizon_s": primary_horizon,
                     "stage": "test", **result, "hac_se": hac["se_hac"], "hac_p_value": hac["p_value_hac"]})

    part = predictions[(predictions["model"] == primary_model) & (predictions["horizon_s"] == primary_horizon)].sort_values("timestamp")
    if not part.empty:
        dev_target = pd.to_numeric(development[f"target_return_{primary_horizon}s"], errors="coerce").abs().dropna()
        cutoff = float(dev_target.quantile(float(config["robustness"]["extreme_abs_target_quantile"]))) if len(dev_target) else np.nan
        masks = {"all": np.ones(len(part), dtype=bool), "extreme_removed": part["actual"].abs().le(cutoff).to_numpy()}
        midpoint = len(part) // 2
        masks["first_half"] = np.arange(len(part)) < midpoint
        masks["second_half"] = np.arange(len(part)) >= midpoint
        for name, mask in masks.items():
            robustness.append({"model": primary_model, "horizon_s": primary_horizon, "sample": name,
                               "extreme_cutoff_frozen_on_development": cutoff,
                               **regression_metrics(part.loc[mask, "actual"], part.loc[mask, "prediction"])})
    return pd.DataFrame(rows), pd.DataFrame(robustness)


def _verdict(
    quality: dict[str, Any], metrics: pd.DataFrame, fold_metrics: pd.DataFrame,
    quantiles: pd.DataFrame, trading: pd.DataFrame, concentration: pd.DataFrame,
    ablation: pd.DataFrame, uncertainty: pd.DataFrame,
    *, primary_model: str, primary_horizon: int, config: dict[str, Any],
) -> dict[str, Any]:
    rules = config["verdict"]
    processed = quality.get("processed", {})
    duration_h = float(processed.get("duration_s") or 0) / 3600
    primary = metrics[(metrics["metric_scope"] == "test") & (metrics["model"] == primary_model) & (metrics["horizon_s"] == primary_horizon)]
    p = primary.iloc[0] if not primary.empty else pd.Series(dtype=object)
    folds = fold_metrics[(fold_metrics["metric_scope"] == "fold") & (fold_metrics["model"] == primary_model) & (fold_metrics["horizon_s"] == primary_horizon)]
    q = quantiles[(quantiles.get("scope") == "test") & (quantiles.get("model") == primary_model) & (quantiles.get("horizon_s") == primary_horizon)]
    t = trading[(trading.get("stage") == "test") & (trading.get("model") == primary_model) & (trading.get("horizon_s") == primary_horizon)
                & (trading.get("execution") == "taker") & np.isclose(trading.get("cost_bps", np.nan), float(rules["confirmatory_incremental_cost_bps"]))]
    c = concentration[(concentration.get("model") == primary_model) & (concentration.get("horizon_s") == primary_horizon)]
    a = ablation[(ablation.get("metric_scope") == "test") & (ablation.get("horizon_s") == primary_horizon)]
    ci = uncertainty[(uncertainty.get("quantity") == "spearman_ic") & (uncertainty.get("model") == primary_model) & (uncertainty.get("horizon_s") == primary_horizon)]
    rank_ic = _finite(p.get("spearman")); fold_values = pd.to_numeric(folds["spearman"], errors="coerce").dropna()
    fold_same = float(np.mean(np.sign(fold_values) == np.sign(rank_ic))) if len(fold_values) and rank_ic != 0 else np.nan
    quantile_rho = _finite(q["bucket_rank_rho"].iloc[0]) if not q.empty and "bucket_rank_rho" in q else np.nan
    top_share = _finite(c.iloc[0].get("top_10pct_share_total_pnl")) if not c.empty else np.nan
    best_hour = _finite(c.iloc[0].get("best_hour_share_total_pnl")) if not c.empty else np.nan
    group_values = dict(zip(a.get("feature_group", []), pd.to_numeric(a.get("spearman", []), errors="coerce")))
    lift = group_values.get("all", np.nan) - group_values.get("price_history", np.nan)
    ci_lower = _finite(ci.iloc[0].get("ci_lower")) if not ci.empty else np.nan
    net_pnl = _finite(t.iloc[0].get("net_pnl")) if not t.empty else np.nan
    test_n = int(_finite(p.get("n"))) if np.isfinite(_finite(p.get("n"))) else 0
    flags = {
        "adequate_duration": duration_h >= float(rules["minimum_research_hours"]),
        "adequate_test_rows": test_n >= int(rules["minimum_final_test_rows"]),
        "data_quality_not_bad": quality.get("status") != "BAD",
        "primary_rank_ic": rank_ic >= float(rules["minimum_primary_test_ic"]),
        "fold_same_sign": np.isfinite(fold_same) and fold_same >= float(rules["minimum_fold_same_sign_fraction"]),
        "monotone_quantiles": np.isfinite(quantile_rho) and quantile_rho >= float(rules["minimum_quantile_rank_correlation"]),
        "not_trade_concentrated": np.isfinite(top_share) and top_share <= float(rules["maximum_top_10pct_pnl_fraction"]),
        "not_hour_concentrated": np.isfinite(best_hour) and best_hour <= float(rules["maximum_best_hour_pnl_fraction"]),
        "microstructure_lift": np.isfinite(lift) and lift >= float(rules["minimum_microstructure_ic_lift"]),
        "positive_ic_ci_lower": np.isfinite(ci_lower) and ci_lower > 0,
        "positive_confirmatory_cost_pnl": np.isfinite(net_pnl) and net_pnl > 0,
    }
    sufficient = flags["adequate_duration"] and flags["adequate_test_rows"] and flags["data_quality_not_bad"]
    statistical = flags["primary_rank_ic"] and flags["fold_same_sign"] and flags["monotone_quantiles"]
    distributed = flags["not_trade_concentrated"] and flags["not_hour_concentrated"]
    if not sufficient:
        verdict = "NO-GO — INSUFFICIENT EVIDENCE (SMOKE ONLY)"
        summary = "The pipeline ran, but the capture is too short, too sparse, or below the predeclared quality gate. No alpha inference is permitted."
        reason_codes = [
            code for flag, code in (
                ("adequate_duration", "INSUFFICIENT_DURATION"),
                ("adequate_test_rows", "INSUFFICIENT_TEST_ROWS"),
                ("data_quality_not_bad", "DATA_QUALITY_BAD"),
            ) if not flags[flag]
        ]
    elif statistical and distributed and flags["microstructure_lift"] and flags["positive_ic_ci_lower"] and flags["positive_confirmatory_cost_pnl"]:
        verdict, summary, reason_codes = "GO", "The predeclared primary signal passed stability, concentration, microstructure-lift, uncertainty, and non-zero-cost screens.", []
    elif statistical and distributed and flags["microstructure_lift"]:
        verdict = "WEAK GO"
        summary = "Stable test structure is present, but uncertainty or the non-zero-cost diagnostic remains weak."
        reason_codes = [name.upper() for name in ("positive_ic_ci_lower", "positive_confirmatory_cost_pnl") if not flags[name]]
    else:
        verdict = "NO-GO"
        summary = "The predeclared primary signal did not jointly survive the required out-of-sample stability, monotonicity, concentration, and microstructure-lift screens."
        reason_codes = [name.upper() for name in ("primary_rank_ic", "fold_same_sign", "monotone_quantiles", "not_trade_concentrated", "not_hour_concentrated", "microstructure_lift") if not flags[name]]
    return _jsonable({
        "verdict": verdict, "summary": summary, "reason_codes": reason_codes,
        "primary_model": primary_model, "primary_horizon_s": primary_horizon,
        "flags": flags,
        "values": {"duration_hours": duration_h, "test_rows": test_n, "test_rank_ic": rank_ic,
                   "fold_same_sign_fraction": fold_same, "quantile_rank_correlation": quantile_rho,
                   "top_10pct_pnl_fraction": top_share, "best_hour_pnl_fraction": best_hour,
                   "microstructure_ic_lift": lift, "ic_ci_lower": ci_lower,
                   "confirmatory_cost_net_pnl": net_pnl},
        "thresholds": rules,
    })


def analyze_dataset(
    config: dict[str, Any], dataset_path: str | Path, *, capture: str | None = None,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Run the complete analysis once and return ``(run_dir, verdict)``."""
    started = time.monotonic()

    def notify(step: int, message: str, **details: Any) -> None:
        if progress:
            progress({"phase": "analyze", "step": step, "total": 12, "message": message,
                      "elapsed_s": time.monotonic() - started, **details})

    notify(1, "Загрузка processed research table")
    dataset_path = Path(dataset_path).resolve()
    capture = capture or dataset_path.parent.name
    research = pd.read_parquet(dataset_path).sort_values("timestamp", kind="stable").reset_index(drop=True)
    research["timestamp"] = pd.to_datetime(research["timestamp"], utc=True)
    horizons = [int(value) for value in config["targets"]["horizons_s"]]
    max_horizon = max(horizons)
    run_dir = create_unique_dir(
        project_path(config, "runs_dir"), make_run_id(config["project"]["symbol"])
    )
    dump_config(config, run_dir / "config.yaml")

    notify(2, "Аудит качества raw и processed данных", rows=len(research), run_dir=str(run_dir))
    raw = {kind: _read_raw(config, kind, capture) for kind in ("orderbook", "trades", "aux")}
    quality = generate_data_quality_payload(
        orderbook=raw["orderbook"], trades=raw["trades"], aux=raw["aux"], processed=research,
        horizons_s=horizons, missing_interval_s=float(config["data_quality"]["missing_interval_s"]),
        sampling_interval_s=float(config["sampling"]["interval_s"]),
        stale_book_s=float(config["data_quality"]["stale_book_s"]), thresholds=config["data_quality"],
    )
    payloads = data_quality_payloads(quality)
    atomic_write_text(payloads["json"] + "\n", run_dir / "data_quality.json")
    _write_csv(payloads["csv"], run_dir / "data_quality.csv")
    atomic_write_text(payloads["markdown"], run_dir / "data_quality_report.md")

    notify(3, "Хронологические splits, purge, embargo и walk-forward folds")
    outer = make_chronological_split(
        research, train_fraction=float(config["splits"]["train_fraction"]),
        validation_fraction=float(config["splits"]["validation_fraction"]),
        test_fraction=float(config["splits"]["test_fraction"]), max_horizon_s=max_horizon,
        embargo_s=int(config["splits"]["embargo_s"]),
    )
    boundary = pd.Timestamp(outer["validation_test_boundary"])
    development = research[research["timestamp"] < boundary].copy()
    folds, fold_count = _walk_folds(development, config, max_horizon)
    groups = resolve_feature_groups(research)
    features = groups["all"]
    candidates = regression_candidates(config)

    notify(4, "Оценка regression/classification моделей на development folds",
           features=len(features), folds=fold_count)
    fold_metrics, oof = _fold_regression(development, folds, candidates, features, horizons, config)
    classification_fold, selected_classifiers = _fold_classification(development, folds, features, horizons, config)
    selected = {horizon: _choose_by_family(fold_metrics, candidates, horizon) for horizon in horizons}
    # All null controls are always carried into the untouched test; only Ridge,
    # LightGBM and logistic hyperparameters are selected on development folds.
    nulls = [candidate for candidate in candidates if candidate.family == "naive"]
    for families in selected.values():
        families.update({f"baseline_{candidate.name}": candidate for candidate in nulls})
    notify(5, "Фиксация выбранных по validation моделей, порогов, quantiles и regimes")
    thresholds, validation_thresholds = _select_thresholds_and_quantiles(oof, selected, config)
    primary_family = str(config["models"]["primary_regression_model"])
    primary_horizon = int(config["models"]["primary_horizon_s"])
    primary_candidate = selected.get(primary_horizon, {}).get(primary_family)
    if primary_candidate is None:
        primary_candidate = selected.get(primary_horizon, {}).get("ridge") or selected.get(primary_horizon, {}).get("linear")
    if primary_candidate is None:
        raise RuntimeError("no primary model survived development evaluation")
    vol_edges = fit_regime_edges(development[config["regimes"]["volatility_feature"]], config["regimes"]["quantiles"])
    spread_edges = fit_regime_edges(development["spread_bps"], config["regimes"]["spread_quantiles"])
    lock = {
        "created_before_test_evaluation": True,
        "primary_model_family": primary_family, "primary_model_selected": primary_candidate.name,
        "primary_horizon_s": primary_horizon, "features": features,
        "selected_regression": {str(h): {family: asdict(candidate) for family, candidate in families.items()} for h, families in selected.items()},
        "selected_classification": {str(h): asdict(candidate) for h, candidate in selected_classifiers.items()},
        "thresholds_and_quantiles": thresholds,
        "regime_edges": {"volatility": vol_edges.tolist(), "spread": spread_edges.tolist()},
        "split_boundaries": {"train_validation": outer["train_validation_boundary"], "validation_test": outer["validation_test_boundary"],
                             "max_horizon": outer["max_horizon"], "embargo": outer["embargo"], "walk_forward_folds_used": fold_count},
        "seed": int(config["project"]["seed"]), "verdict_rules": config["verdict"],
    }
    atomic_write_json(_jsonable(lock), run_dir / "decision_lock.json")
    atomic_write_text(sha256_file(run_dir / "decision_lock.json") + "\n", run_dir / "decision_lock.sha256")

    notify(6, "Однократная оценка untouched final test")
    test_rows, test_predictions, fitted = _final_regression(research, outer, selected, features, config, max_horizon)
    fold_metrics = pd.concat([fold_metrics, test_rows], ignore_index=True)
    summary = pd.concat([_summary_from_folds(fold_metrics), test_rows], ignore_index=True)
    classification_test = _final_classification(research, outer, selected_classifiers, features, horizons, config, max_horizon)
    classification_all = pd.concat([classification_fold, classification_test], ignore_index=True)
    notify(7, "Расчёт univariate-статистики, HAC inference и FDR")
    feature_stats, correlations, multiple, stability = _univariate_tables(
        research, development, outer, folds, features, horizons, config
    )
    notify(8, "Quantile, cost, concentration и regime diagnostics")
    quantiles, quantile_meta = _quantile_tables(oof, test_predictions, thresholds, research, outer, features, horizons, config)
    trading, costs, trades, concentration, contribution = _trading_tables(test_predictions, thresholds, config)
    regimes, regime_meta = _regime_table(test_predictions, trades, development, config, thresholds)
    notify(9, "Расчёт model importance и feature-group ablations")
    importance, permutation, ablation = _importance_and_ablation(
        research, development, outer, selected, fitted, groups, horizons, config, max_horizon, folds
    )
    notify(10, "Оценка time-series uncertainty, robustness и verdict")
    uncertainty, robustness = _uncertainty_and_robustness(
        test_predictions, trades, development, primary_candidate.name, primary_horizon, config
    )
    verdict = _verdict(
        quality, summary, fold_metrics, quantiles, trading, concentration, ablation, uncertainty,
        primary_model=primary_candidate.name, primary_horizon=primary_horizon, config=config,
    )

    notify(11, "Запись числовых audit-таблиц и манифестов")
    tables = {
        "metrics_summary.csv": summary, "fold_metrics.csv": fold_metrics,
        "feature_statistics.csv": feature_stats, "feature_target_correlations.csv": correlations,
        "feature_stability.csv": stability, "multiple_testing.csv": multiple,
        "quantile_analysis.csv": quantiles, "classification_metrics.csv": classification_all,
        "validation_thresholds.csv": validation_thresholds, "trading_metrics.csv": trading,
        "cost_sensitivity.csv": costs, "regime_metrics.csv": regimes,
        "ablation_metrics.csv": ablation, "bootstrap_confidence_intervals.csv": uncertainty,
        "feature_importance.csv": importance, "permutation_importance.csv": permutation,
        "concentration_metrics.csv": concentration, "contribution_metrics.csv": contribution,
        "robustness_metrics.csv": robustness,
    }
    for name, table in tables.items():
        _write_csv(table, run_dir / name)
    atomic_write_dataframe(test_predictions, run_dir / "predictions_test.parquet")
    atomic_write_dataframe(trades, run_dir / "trades_test.parquet")
    atomic_write_json(_jsonable(summary.to_dict("records")), run_dir / "metrics_summary.json")
    atomic_write_json(_jsonable(quantile_meta), run_dir / "quantile_metadata.json")
    atomic_write_json(_jsonable(regime_meta), run_dir / "regime_edges.json")
    atomic_write_json(verdict, run_dir / "verdict.json")
    atomic_write_json({"groups": groups, "all_features": features}, run_dir / "feature_list.json")
    atomic_write_json(_jsonable(lock["split_boundaries"]), run_dir / "split_boundaries.json")
    environment = environment_manifest(["numpy", "pandas", "pyarrow", "scipy", "scikit-learn", "lightgbm", "matplotlib", "seaborn", "websockets"])
    atomic_write_json(environment, run_dir / "environment.json")
    fingerprint = dataset_fingerprint([dataset_path])
    dataset_manifest = {
        "sha256": fingerprint["sha256"], "capture_session": capture,
        "research_table": str(dataset_path), "rows": len(research),
        "data_interval": {"first_timestamp": research["timestamp"].min(), "last_timestamp": research["timestamp"].max()},
        "duration_s": quality.get("processed", {}).get("duration_s"),
        "book_usable_pct": quality.get("processed", {}).get("book_usable_pct"),
        "fingerprint": fingerprint,
    }
    atomic_write_json(_jsonable(dataset_manifest), run_dir / "dataset_manifest.json")
    atomic_write_json({
        "git_commit": git_commit(PROJECT_ROOT), "seed": config["project"]["seed"],
        "capture_session": capture, "decision_lock_sha256": sha256_file(run_dir / "decision_lock.json"),
        "outputs": sorted(tables),
    }, run_dir / "run_manifest.json")
    recorded_errors = {
        name: table.loc[table["error"].notna(), "error"].astype(str).tolist()
        for name, table in tables.items() if "error" in table and table["error"].notna().any()
    }
    atomic_write_json({"recorded_analysis_errors": recorded_errors}, run_dir / "analysis_warnings.json")
    atomic_write_text((git_commit(PROJECT_ROOT) or "not-a-git-checkout") + "\n", run_dir / "git_commit.txt")

    notify(12, "Построение 20 графиков и Markdown-отчёта")
    generate_plots(
        run_dir, research, feature_columns=features, primary_model=primary_candidate.name,
        primary_horizon_s=primary_horizon,
        confirmatory_cost_bps=float(config["verdict"]["confirmatory_incremental_cost_bps"]),
        dpi=int(config["plots"]["dpi"]),
    )
    render_report(run_dir)
    notify(12, "Анализ завершён", complete=True, run_dir=str(run_dir), verdict=verdict["verdict"])
    return run_dir, verdict


__all__ = ["analyze_dataset", "classification_metrics", "regression_metrics"]
