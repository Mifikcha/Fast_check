"""Stage 2 day-level replication; Stage 1 artifacts remain read-only."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import yaml
from scipy import stats

from .book import OrderBook
from .build_dataset import build_dataset
from .config import PROJECT_ROOT, project_path
from .io_utils import atomic_write_dataframe, atomic_write_json, sha256_file
from .long_collection import completed_utc_days, recording_sessions
from .models import Candidate, build_regressor, resolve_feature_groups
from .statistics import assign_quantile_buckets, bootstrap_correlation, fit_quantile_edges, safe_correlation
from .targets import add_targets


@dataclass
class FrozenBaseline:
    config: dict[str, Any]
    features: list[str]
    model: Any
    original: pd.DataFrame
    original_train: pd.DataFrame
    quantile_edges: np.ndarray
    selectivity_thresholds: dict[float, float]
    group_models: dict[str, Any]
    obi_edges: dict[str, np.ndarray]
    regime_edges: dict[str, np.ndarray]


def _float(value: Any) -> float:
    try:
        result = float(value)
        return result if np.isfinite(result) else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def _frozen_hash(features: list[str]) -> str:
    return hashlib.sha256(json.dumps(features, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def load_frozen_baseline(config_path: Path, root: Path = PROJECT_ROOT) -> FrozenBaseline:
    """Verify source fingerprints, then refit only on the original development interval."""
    frozen = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    run = Path(root) / "runs" / frozen["source_run"]
    lock_path = run / "decision_lock.json"
    if not lock_path.exists() or sha256_file(lock_path) != frozen["source_decision_lock_sha256"]:
        raise ValueError("frozen decision-lock hash does not match the original run")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    features = list(lock["features"])
    if _frozen_hash(features) != frozen["source_feature_list_sha256"]:
        raise ValueError("frozen feature-list hash does not match the original run")
    manifest = json.loads((run / "dataset_manifest.json").read_text(encoding="utf-8"))
    dataset = Path(manifest["research_table"])
    if not dataset.exists() or sha256_file(dataset) != frozen["source_dataset_sha256"]:
        raise ValueError("frozen source-dataset hash does not match the original run")
    selected = lock["selected_regression"]["30"]["ridge"]
    threshold = lock["thresholds_and_quantiles"]["ridge_a0.1|30"]["threshold"]
    if (selected["name"] != frozen["model"] or float(selected["params"]["alpha"]) != float(frozen["alpha"])
            or int(frozen["horizon_s"]) != 30 or float(threshold) != float(frozen["threshold"])):
        raise ValueError("frozen baseline parameters differ from the original decision lock")
    original = pd.read_parquet(dataset)
    original["timestamp"] = pd.to_datetime(original["timestamp"], utc=True)
    boundary = pd.Timestamp(lock["split_boundaries"]["validation_test"]).tz_convert("UTC")
    max_horizon = int(pd.Timedelta(lock["split_boundaries"]["max_horizon"]).total_seconds())
    train = original.loc[
        (original["timestamp"] + pd.Timedelta(seconds=max_horizon) < boundary)
        & original["usable"].fillna(False).astype(bool)
        & original["target_valid_30s"].fillna(False).astype(bool)
        & original["target_path_valid_30s"].fillna(False).astype(bool)
        & np.isfinite(original["target_return_30s"])
    ].copy()
    if train.empty or set(features) - set(train):
        raise ValueError("original development rows or frozen features are unavailable")
    model = build_regressor(Candidate(frozen["model"], "ridge", {"alpha": float(frozen["alpha"])}), seed=10072, n_jobs=1)
    model.fit(train[features], train["target_return_30s"])
    edges = np.asarray([(-np.inf if i == 0 else np.inf if i == 10 else x)
                        for i, x in enumerate(lock["thresholds_and_quantiles"]["ridge_a0.1|30"]["quantile_edges"])], dtype=float)
    prior_predictions = model.predict(train[features])
    tails = [float(x) for x in frozen["stage2"]["selectivity_tail_percent"]]
    selectivity = {p: float(np.quantile(np.abs(prior_predictions), 1 - p / 100)) for p in tails}
    groups = resolve_feature_groups(original)
    group_models = {}
    for name in ("price_history", "trade_flow", "order_book", "order_book_trade_flow", "all"):
        cols = groups[name]
        if cols:
            candidate = build_regressor(Candidate(f"ridge_{name}", "ridge", {"alpha": float(frozen["alpha"])}), seed=10072, n_jobs=1)
            candidate.fit(train[cols], train["target_return_30s"])
            group_models[name] = (cols, candidate)
    obi_features = ["obi_1", "obi_5", "obi_10", "obi_20", "microdev"]
    obi_edges = {name: fit_quantile_edges(train[name], 10) for name in obi_features}
    regime_edges = {
        "volatility": np.quantile(train["realized_vol_60s"].dropna(), [1 / 3, 2 / 3]),
        "spread": np.quantile(train["spread_bps"].dropna(), [1 / 3, 2 / 3]),
    }
    return FrozenBaseline(frozen, features, model, original, train, edges, selectivity, group_models, obi_edges, regime_edges)


def _intervals_from_mask(times: pd.DatetimeIndex, mask: np.ndarray, reason: str) -> pd.DataFrame:
    if not len(mask):
        return pd.DataFrame(columns=["start", "end", "reason", "duration_s"])
    starts = np.flatnonzero(mask & ~np.r_[False, mask[:-1]])
    ends = np.flatnonzero(mask & ~np.r_[mask[1:], False])
    return pd.DataFrame({"start": times[starts], "end": times[ends], "reason": reason,
                         "duration_s": ends - starts + 1})


def buffered_eligibility(
    times: Iterable[Any], source_invalid: pd.Series | np.ndarray, *,
    before_s: int, after_s: int, horizons_s: Iterable[int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Exclude buffered defects and any label path touching one, including endpoints."""
    index = pd.DatetimeIndex(pd.to_datetime(times, utc=True))
    if index.has_duplicates or not index.is_monotonic_increasing:
        raise ValueError("timestamps must be unique and monotonic")
    raw = np.asarray(source_invalid, dtype=bool)
    if len(raw) != len(index) or before_s < 0 or after_s < 0:
        raise ValueError("invalid mask length or exclusion buffer")
    intervals = _intervals_from_mask(index, raw, "book_or_collection_invalid")
    buffered = np.zeros(len(index), dtype=bool)
    for row in intervals.itertuples():
        left = index.searchsorted(row.start - pd.Timedelta(seconds=before_s), side="left")
        right = index.searchsorted(row.end + pd.Timedelta(seconds=after_s), side="right")
        buffered[left:right] = True
    intervals["buffered_start"] = intervals["start"] - pd.Timedelta(seconds=before_s)
    intervals["buffered_end"] = intervals["end"] + pd.Timedelta(seconds=after_s)
    result = pd.DataFrame({"timestamp": index, "source_invalid": raw, "buffered_invalid": buffered})
    cumulative = np.r_[0, np.cumsum(buffered.astype(int))]
    ns = index.asi8
    for horizon in horizons_s:
        target = ns + int(horizon) * 1_000_000_000
        exit_index = np.searchsorted(ns, target, side="left")
        inside = exit_index < len(index)
        exact = np.zeros(len(index), dtype=bool)
        exact[inside] = ns[exit_index[inside]] == target[inside]
        count = np.zeros(len(index), dtype=int)
        count[inside] = cumulative[exit_index[inside] + 1] - cumulative[np.flatnonzero(inside)]
        result[f"eligible_{int(horizon)}s"] = exact & (count == 0)
    return result, intervals


def nonoverlap_ic_by_phase(frame: pd.DataFrame, *, horizon_s: int,
                           signal_col: str = "prediction", target_col: str = "actual") -> pd.DataFrame:
    """Enumerate every absolute UTC phase; a phase is never selected by its IC."""
    if horizon_s <= 0:
        raise ValueError("horizon_s must be positive")
    timestamps = pd.to_datetime(frame["timestamp"], utc=True)
    phase = (timestamps.astype("int64") // 1_000_000_000) % horizon_s
    signal = pd.to_numeric(frame[signal_col], errors="coerce")
    target = pd.to_numeric(frame[target_col], errors="coerce")
    valid = np.isfinite(signal) & np.isfinite(target)
    rows = []
    for offset in range(horizon_s):
        use = valid & phase.eq(offset)
        x, y = signal.loc[use], target.loc[use]
        rows.append({"horizon_s": horizon_s, "phase": offset, "n": len(x),
                     "pearson": safe_correlation(x, y, "pearson"),
                     "spearman": safe_correlation(x, y, "spearman")})
    return pd.DataFrame(rows)


def stale_book_at(ticks: pd.DataFrame, decision: pd.Timestamp, *, latency_ms: int,
                  max_age_ms: int = 2_000) -> dict[str, Any] | None:
    deadline = pd.Timestamp(decision) - pd.Timedelta(milliseconds=latency_ms)
    timestamps = pd.DatetimeIndex(pd.to_datetime(ticks["timestamp"], utc=True))
    index = timestamps.searchsorted(deadline, side="right") - 1
    if index < 0 or (deadline - timestamps[index]).total_seconds() * 1000 > max_age_ms:
        return None
    row = ticks.iloc[index]
    return row.to_dict() if bool(row["book_valid"]) else None


def quote_at_or_after(ticks: pd.DataFrame, decision: pd.Timestamp, *, latency_ms: int,
                      max_wait_ms: int = 250) -> dict[str, Any] | None:
    deadline = pd.Timestamp(decision) + pd.Timedelta(milliseconds=latency_ms)
    ticks = ticks.loc[ticks["book_valid"]]
    timestamps = pd.DatetimeIndex(pd.to_datetime(ticks["timestamp"], utc=True))
    index = timestamps.searchsorted(deadline, side="left")
    if index >= len(ticks) or (timestamps[index] - deadline).total_seconds() * 1000 > max_wait_ms:
        return None
    row = ticks.iloc[index]
    return row.to_dict() if bool(row["book_valid"]) else None


__all__ = ["FrozenBaseline", "buffered_eligibility", "load_frozen_baseline", "nonoverlap_ic_by_phase", "quote_at_or_after", "stale_book_at"]
