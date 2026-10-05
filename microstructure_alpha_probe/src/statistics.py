from __future__ import annotations

import math
import re
from collections.abc import Callable, Iterable, Sequence
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_selection import mutual_info_regression


ArrayLike = Sequence[float] | np.ndarray | pd.Series


def _float_array(values: ArrayLike) -> np.ndarray:
    return pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)


def finite_pairs(x: ArrayLike, y: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """Return pairwise-complete finite one-dimensional arrays."""
    xa, ya = _float_array(x), _float_array(y)
    if xa.shape != ya.shape:
        raise ValueError("x and y must have the same shape")
    keep = np.isfinite(xa) & np.isfinite(ya)
    return xa[keep], ya[keep]


def safe_correlation(x: ArrayLike, y: ArrayLike, method: str = "pearson") -> float:
    """Correlation with pairwise deletion; constants return NaN, never zero."""
    xa, ya = finite_pairs(x, y)
    if len(xa) < 2 or np.ptp(xa) == 0 or np.ptp(ya) == 0:
        return float("nan")
    if method == "pearson":
        return float(np.corrcoef(xa, ya)[0, 1])
    if method == "spearman":
        return float(stats.spearmanr(xa, ya).statistic)
    raise ValueError("method must be 'pearson' or 'spearman'")


def mutual_information(
    x: ArrayLike,
    y: ArrayLike,
    *,
    max_rows: int = 50_000,
    seed: int = 42,
) -> tuple[float, int]:
    """Deterministic univariate MI; returns (estimate, rows actually used)."""
    xa, ya = finite_pairs(x, y)
    if len(xa) < 4 or np.ptp(xa) == 0 or np.ptp(ya) == 0:
        return float("nan"), int(len(xa))
    if max_rows <= 0:
        raise ValueError("max_rows must be positive")
    if len(xa) > max_rows:
        rng = np.random.default_rng(seed)
        take = np.sort(rng.choice(len(xa), size=max_rows, replace=False))
        xa, ya = xa[take], ya[take]
    value = mutual_info_regression(
        xa.reshape(-1, 1), ya, random_state=seed, discrete_features=False
    )[0]
    return float(value), int(len(xa))


def _normal_interval(estimate: float, se: float, confidence: float) -> tuple[float, float]:
    critical = float(stats.norm.ppf(0.5 + confidence / 2))
    return estimate - critical * se, estimate + critical * se


def hac_mean_inference(
    values: ArrayLike,
    *,
    max_lag: int,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Newey-West inference for the mean using Bartlett weights.

    The input must already represent the intended regularly spaced estimand.
    Non-finite values are removed and their count remains visible in ``n``.
    """
    z = _float_array(values)
    z = z[np.isfinite(z)]
    n, q = len(z), int(max_lag)
    result: dict[str, Any] = {
        "estimate": float(np.mean(z)) if n else float("nan"),
        "se_hac": float("nan"),
        "t_hac": float("nan"),
        "p_value_hac": float("nan"),
        "ci_lower": float("nan"),
        "ci_upper": float("nan"),
        "n": int(n),
        "max_lag": q,
        "reason": None,
    }
    if q < 0:
        raise ValueError("max_lag cannot be negative")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between zero and one")
    if n <= q + 2:
        result["reason"] = "n_not_greater_than_max_lag_plus_2"
        return result

    centered = z - result["estimate"]
    gamma0 = float(np.dot(centered, centered) / n)
    lrv = gamma0
    for lag in range(1, q + 1):
        gamma = float(np.dot(centered[lag:], centered[:-lag]) / n)
        lrv += 2.0 * (1.0 - lag / (q + 1.0)) * gamma

    tolerance = np.finfo(float).eps * max(1.0, gamma0) * 100
    if lrv < -tolerance:
        result["reason"] = "negative_long_run_variance"
        return result
    lrv = max(0.0, lrv)
    se = math.sqrt(lrv / n)
    if not np.isfinite(se) or se <= 0:
        result["reason"] = "zero_or_invalid_hac_standard_error"
        return result

    estimate = float(result["estimate"])
    t_value = estimate / se
    lower, upper = _normal_interval(estimate, se, confidence)
    result.update(
        se_hac=float(se),
        t_hac=float(t_value),
        p_value_hac=float(2 * stats.norm.sf(abs(t_value))),
        ci_lower=float(lower),
        ci_upper=float(upper),
    )
    return result


def _hac_standardized_slope(x: np.ndarray, y: np.ndarray, max_lag: int) -> dict[str, Any]:
    """OLS slope and Newey-West sandwich SE after standardizing x and y."""
    n, q = len(x), int(max_lag)
    output: dict[str, Any] = {
        "correlation": float("nan"),
        "se_hac": float("nan"),
        "t_hac": float("nan"),
        "p_value_hac": float("nan"),
        "n": int(n),
        "max_lag": q,
        "reason": None,
    }
    if q < 0:
        raise ValueError("max_lag cannot be negative")
    if n <= q + 2:
        output["reason"] = "n_not_greater_than_max_lag_plus_2"
        return output
    x_sd, y_sd = float(np.std(x)), float(np.std(y))
    x_scale = max(1.0, float(np.max(np.abs(x))))
    y_scale = max(1.0, float(np.max(np.abs(y))))
    tolerance = np.finfo(float).eps * 100
    if (
        x_sd <= tolerance * x_scale
        or y_sd <= tolerance * y_scale
        or not np.isfinite(x_sd + y_sd)
    ):
        output["reason"] = "constant_or_invalid_series"
        return output

    xs = (x - np.mean(x)) / x_sd
    ys = (y - np.mean(y)) / y_sd
    design = np.column_stack((np.ones(n), xs))
    try:
        xtx_inv = np.linalg.inv(design.T @ design)
    except np.linalg.LinAlgError:
        output["reason"] = "singular_standardized_design"
        return output
    beta = xtx_inv @ design.T @ ys
    residual = ys - design @ beta
    scores = design * residual[:, None]
    meat = scores.T @ scores
    for lag in range(1, q + 1):
        cross = scores[lag:].T @ scores[:-lag]
        weight = 1.0 - lag / (q + 1.0)
        meat += weight * (cross + cross.T)
    covariance = xtx_inv @ meat @ xtx_inv
    variance = float(covariance[1, 1])
    if variance < -np.finfo(float).eps * 100:
        output["reason"] = "negative_hac_slope_variance"
        return output
    se = math.sqrt(max(0.0, variance))
    coefficient = float(beta[1])
    output["correlation"] = coefficient
    if se <= 0 or not np.isfinite(se):
        output["reason"] = "zero_or_invalid_hac_standard_error"
        return output
    t_value = coefficient / se
    output.update(
        se_hac=float(se),
        t_hac=float(t_value),
        p_value_hac=float(2 * stats.norm.sf(abs(t_value))),
    )
    return output


def hac_correlation_inference(
    x: ArrayLike,
    y: ArrayLike,
    *,
    max_lag: int,
    method: str = "pearson",
) -> dict[str, Any]:
    """HAC slope inference for Pearson or rank-transformed Spearman IC."""
    xa, ya = finite_pairs(x, y)
    if method == "spearman":
        xa, ya = stats.rankdata(xa), stats.rankdata(ya)
    elif method != "pearson":
        raise ValueError("method must be 'pearson' or 'spearman'")
    result = _hac_standardized_slope(xa, ya, max_lag)
    if result["reason"] is None or result["reason"] == "zero_or_invalid_hac_standard_error":
        if len(xa) >= 2 and np.ptp(xa) > 0 and np.ptp(ya) > 0:
            test = stats.pearsonr(xa, ya)
            result["p_value_iid_approx"] = float(test.pvalue)
        else:
            result["p_value_iid_approx"] = float("nan")
    else:
        result["p_value_iid_approx"] = float("nan")
    return result


def _target_horizon_seconds(name: str) -> int | None:
    match = re.search(r"_(\d+)s(?:$|_)", name)
    return int(match.group(1)) if match else None


def feature_target_statistics(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    target_columns: Sequence[str],
    *,
    sampling_interval_s: int = 1,
    minimum_hac_lag: int = 0,
    max_mi_rows: int = 50_000,
    seed: int = 42,
    stage: str = "unspecified",
) -> pd.DataFrame:
    """Pairwise Pearson/Spearman/MI table with IID-labelled and HAC p-values."""
    if sampling_interval_s <= 0:
        raise ValueError("sampling_interval_s must be positive")
    missing = (set(feature_columns) | set(target_columns)) - set(frame.columns)
    if missing:
        raise KeyError(f"Missing columns: {sorted(missing)}")

    rows: list[dict[str, Any]] = []
    for target in target_columns:
        horizon = _target_horizon_seconds(target)
        horizon_rows = math.ceil(horizon / sampling_interval_s) if horizon else 1
        hac_lag = max(int(minimum_hac_lag), horizon_rows - 1)
        for feature in feature_columns:
            x, y = finite_pairs(frame[feature], frame[target])
            pearson = hac_correlation_inference(x, y, max_lag=hac_lag, method="pearson")
            spearman = hac_correlation_inference(x, y, max_lag=hac_lag, method="spearman")
            mi, mi_n = mutual_information(x, y, max_rows=max_mi_rows, seed=seed)
            rows.append(
                {
                    "stage": stage,
                    "feature": feature,
                    "target": target,
                    "horizon_s": horizon,
                    "n": int(len(x)),
                    "pearson": pearson["correlation"],
                    "pearson_p_iid_approx": pearson["p_value_iid_approx"],
                    "pearson_hac_se": pearson["se_hac"],
                    "pearson_hac_t": pearson["t_hac"],
                    "pearson_p_hac": pearson["p_value_hac"],
                    "spearman": spearman["correlation"],
                    "spearman_p_iid_approx": spearman["p_value_iid_approx"],
                    "spearman_hac_se": spearman["se_hac"],
                    "spearman_hac_t": spearman["t_hac"],
                    "spearman_p_hac": spearman["p_value_hac"],
                    "mutual_information": mi,
                    "mutual_information_n": mi_n,
                    "hac_max_lag": hac_lag,
                    "correlation_sign": (
                        int(np.sign(spearman["correlation"]))
                        if np.isfinite(spearman["correlation"])
                        else np.nan
                    ),
                }
            )
    return pd.DataFrame(rows)


def benjamini_hochberg(
    p_values: Iterable[float], alpha: float = 0.05
) -> tuple[np.ndarray, np.ndarray]:
    """Benjamini-Hochberg q-values; NaNs are excluded from m and preserved."""
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between zero and one")
    p = _float_array(list(p_values))
    finite = np.isfinite(p)
    if np.any((p[finite] < 0) | (p[finite] > 1)):
        raise ValueError("finite p-values must lie in [0, 1]")
    q = np.full_like(p, np.nan)
    reject = np.zeros(len(p), dtype=bool)
    finite_indices = np.flatnonzero(finite)
    m = len(finite_indices)
    if not m:
        return q, reject
    order = finite_indices[np.argsort(p[finite_indices], kind="mergesort")]
    adjusted_sorted = p[order] * m / np.arange(1, m + 1)
    adjusted_sorted = np.minimum.accumulate(adjusted_sorted[::-1])[::-1]
    adjusted_sorted = np.minimum(1.0, adjusted_sorted)
    q[order] = adjusted_sorted
    reject[finite_indices] = q[finite_indices] <= alpha
    return q, reject


def apply_bh_fdr(
    frame: pd.DataFrame,
    *,
    p_column: str,
    alpha: float = 0.05,
    family_columns: Sequence[str] = (),
    q_column: str = "q_value",
    reject_column: str = "reject_fdr",
) -> pd.DataFrame:
    """Apply BH within explicit families while retaining every original row."""
    if p_column not in frame:
        raise KeyError(p_column)
    missing = set(family_columns) - set(frame.columns)
    if missing:
        raise KeyError(f"Missing family columns: {sorted(missing)}")
    out = frame.copy()
    out[q_column] = np.nan
    out[reject_column] = False
    groups = [((), out.index)] if not family_columns else out.groupby(
        list(family_columns), dropna=False, sort=False
    ).groups.items()
    for _, index in groups:
        q, reject = benjamini_hochberg(out.loc[index, p_column], alpha=alpha)
        out.loc[index, q_column] = q
        out.loc[index, reject_column] = reject
    return out


def autocorrelation(values: ArrayLike, lag: int) -> tuple[float, int]:
    """Time-position-preserving pairwise autocorrelation at a positive row lag."""
    if lag <= 0:
        raise ValueError("lag must be positive")
    x = _float_array(values)
    if len(x) <= lag:
        return float("nan"), 0
    left, right = finite_pairs(x[:-lag], x[lag:])
    return safe_correlation(left, right), int(len(left))


def descriptive_statistics(
    frame: pd.DataFrame,
    feature_columns: Sequence[str] | None = None,
    *,
    autocorrelation_lags: Sequence[int] = (1, 5, 10, 30, 60),
    near_zero_relative_tolerance: float = 1e-12,
) -> pd.DataFrame:
    """Requested feature distribution table, treating non-finite values as missing."""
    columns = list(feature_columns or frame.select_dtypes(include=[np.number]).columns)
    missing = set(columns) - set(frame.columns)
    if missing:
        raise KeyError(f"Missing columns: {sorted(missing)}")
    quantiles = (0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)
    rows: list[dict[str, Any]] = []
    total = len(frame)
    for name in columns:
        raw = _float_array(frame[name])
        values = raw[np.isfinite(raw)]
        n = len(values)
        mean = float(np.mean(values)) if n else float("nan")
        std = float(np.std(values, ddof=1)) if n >= 2 else float("nan")
        unique = int(len(np.unique(values)))
        scale = max(1.0, abs(mean)) if np.isfinite(mean) else 1.0
        stable_variance = bool(np.isfinite(std) and std > near_zero_relative_tolerance * scale)
        row: dict[str, Any] = {
            "feature": name,
            "count": int(n),
            "missing_pct": float(100 * (total - n) / total) if total else float("nan"),
            "mean": mean,
            "std": std,
            "min": float(np.min(values)) if n else float("nan"),
            "max": float(np.max(values)) if n else float("nan"),
            "skewness": float(stats.skew(values, bias=False)) if n >= 3 and stable_variance else float("nan"),
            "excess_kurtosis": (
                float(stats.kurtosis(values, fisher=True, bias=False))
                if n >= 4 and stable_variance
                else float("nan")
            ),
            "nunique": unique,
            "near_zero_variance": bool(
                unique <= 1
                or not stable_variance
            ),
            "near_zero_variance_rule": (
                f"nunique<=1 or sample_std<={near_zero_relative_tolerance:g}*max(1,abs(mean))"
            ),
        }
        for probability in quantiles:
            label = "median" if probability == 0.5 else f"p{int(probability * 100)}"
            row[label] = float(np.quantile(values, probability)) if n else float("nan")
        for lag in autocorrelation_lags:
            value, pairs = autocorrelation(raw, int(lag))
            row[f"autocorr_lag_{lag}"] = value
            row[f"autocorr_n_lag_{lag}"] = pairs
        rows.append(row)
    return pd.DataFrame(rows)


def fold_ic_stability(
    fold_metrics: pd.DataFrame,
    *,
    group_columns: Sequence[str] = ("feature", "target"),
    ic_column: str = "spearman",
    full_ic_column: str | None = None,
) -> pd.DataFrame:
    """Summarize IC over folds; the number of folds, not rows, is the t-stat unit."""
    required = set(group_columns) | {ic_column}
    if full_ic_column:
        required.add(full_ic_column)
    missing = required - set(fold_metrics.columns)
    if missing:
        raise KeyError(f"Missing columns: {sorted(missing)}")
    rows: list[dict[str, Any]] = []
    grouped: Any = fold_metrics.groupby(list(group_columns), dropna=False, sort=False)
    for keys, part in grouped:
        values = _float_array(part[ic_column])
        values = values[np.isfinite(values)]
        k = len(values)
        mean = float(np.mean(values)) if k else float("nan")
        sd = float(np.std(values, ddof=1)) if k >= 2 else float("nan")
        t_value = mean / (sd / math.sqrt(k)) if k >= 3 and sd > 0 else float("nan")
        row = dict(zip(group_columns, keys if isinstance(keys, tuple) else (keys,)))
        reference = (
            float(pd.to_numeric(part[full_ic_column], errors="coerce").dropna().iloc[0])
            if full_ic_column and pd.to_numeric(part[full_ic_column], errors="coerce").notna().any()
            else mean
        )
        row.update(
            fold_count=int(k),
            mean_ic=mean,
            median_ic=float(np.median(values)) if k else float("nan"),
            ic_std=sd,
            ic_t_stat=t_value,
            fraction_positive=float(np.mean(values > 0)) if k else float("nan"),
            fraction_same_sign=(
                float(np.mean(np.sign(values) == np.sign(reference)))
                if k and np.isfinite(reference) and reference != 0
                else float("nan")
            ),
            worst_ic=float(np.min(values)) if k else float("nan"),
            best_ic=float(np.max(values)) if k else float("nan"),
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _contiguous_runs(valid: np.ndarray, segment_ids: np.ndarray | None) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index, is_valid in enumerate(valid):
        same_segment = True
        if index and segment_ids is not None:
            left, right = segment_ids[index - 1], segment_ids[index]
            same_segment = bool(pd.notna(left) and pd.notna(right) and left == right)
        if is_valid and (index == 0 or (valid[index - 1] and same_segment)):
            if start is None:
                start = index
        elif is_valid:
            if start is not None:
                runs.append((start, index))
            start = index
        elif start is not None:
            runs.append((start, index))
            start = None
    if start is not None:
        runs.append((start, len(valid)))
    return runs


def moving_block_indices(
    n: int,
    block_length: int,
    *,
    rng: np.random.Generator,
    valid_mask: Sequence[bool] | None = None,
    segment_ids: Sequence[Any] | None = None,
    sample_size: int | None = None,
) -> np.ndarray:
    """Sample adjacent blocks without crossing invalid rows or segment boundaries."""
    if n < 0 or block_length <= 0:
        raise ValueError("n must be non-negative and block_length positive")
    valid = np.ones(n, dtype=bool) if valid_mask is None else np.asarray(valid_mask, dtype=bool)
    if len(valid) != n:
        raise ValueError("valid_mask length mismatch")
    segments = None if segment_ids is None else np.asarray(segment_ids, dtype=object)
    if segments is not None and len(segments) != n:
        raise ValueError("segment_ids length mismatch")
    target = int(np.sum(valid)) if sample_size is None else int(sample_size)
    if target <= 0:
        return np.array([], dtype=np.int64)
    starts: list[int] = []
    for run_start, run_end in _contiguous_runs(valid, segments):
        if run_end - run_start >= block_length:
            starts.extend(range(run_start, run_end - block_length + 1))
    if not starts:
        raise ValueError("no contiguous valid run is as long as block_length")
    starts_array = np.asarray(starts, dtype=np.int64)
    chunks: list[np.ndarray] = []
    sampled = 0
    while sampled < target:
        start = int(rng.choice(starts_array))
        chunk = np.arange(start, start + block_length, dtype=np.int64)
        chunks.append(chunk)
        sampled += len(chunk)
    return np.concatenate(chunks)[:target]


def moving_block_bootstrap(
    *arrays: ArrayLike,
    statistic: Callable[..., float],
    block_length: int,
    n_resamples: int = 1_000,
    confidence: float = 0.95,
    seed: int = 42,
    valid_mask: Sequence[bool] | None = None,
    segment_ids: Sequence[Any] | None = None,
    return_distribution: bool = False,
) -> dict[str, Any]:
    """Deterministic moving-block percentile CI for paired chronological arrays."""
    if not arrays:
        raise ValueError("at least one array is required")
    converted = [np.asarray(a) for a in arrays]
    n = len(converted[0])
    if any(len(a) != n for a in converted):
        raise ValueError("all arrays must have equal length")
    if n_resamples <= 0 or not 0 < confidence < 1:
        raise ValueError("invalid bootstrap parameters")

    valid = np.ones(n, dtype=bool)
    if valid_mask is not None:
        supplied = np.asarray(valid_mask, dtype=bool)
        if len(supplied) != n:
            raise ValueError("valid_mask length mismatch")
        valid &= supplied
    for array in converted:
        numeric = pd.to_numeric(pd.Series(array), errors="coerce").to_numpy(dtype=float)
        valid &= np.isfinite(numeric)
    n_valid = int(np.sum(valid))
    result: dict[str, Any] = {
        "estimate": float("nan"),
        "ci_lower": float("nan"),
        "ci_upper": float("nan"),
        "n": n_valid,
        "block_length": int(block_length),
        "n_resamples": int(n_resamples),
        "n_valid_resamples": 0,
        "confidence": float(confidence),
        "exploratory": bool(n_valid < 5 * block_length),
        "reason": None,
    }
    if n_valid == 0:
        result["reason"] = "no_finite_observations"
        return result
    try:
        estimate = float(statistic(*(a[valid] for a in converted)))
    except (TypeError, ValueError, FloatingPointError):
        estimate = float("nan")
    result["estimate"] = estimate

    rng = np.random.default_rng(seed)
    replicates: list[float] = []
    try:
        for _ in range(n_resamples):
            indices = moving_block_indices(
                n,
                block_length,
                rng=rng,
                valid_mask=valid,
                segment_ids=segment_ids,
                sample_size=n_valid,
            )
            value = float(statistic(*(a[indices] for a in converted)))
            if np.isfinite(value):
                replicates.append(value)
    except ValueError as error:
        result["reason"] = str(error)
        return result

    result["n_valid_resamples"] = len(replicates)
    if not replicates:
        result["reason"] = "no_finite_bootstrap_replicates"
        return result
    tail = (1.0 - confidence) / 2.0
    result["ci_lower"] = float(np.quantile(replicates, tail))
    result["ci_upper"] = float(np.quantile(replicates, 1.0 - tail))
    if return_distribution:
        result["distribution"] = np.asarray(replicates)
    return result


def bootstrap_correlation(
    signal: ArrayLike,
    target: ArrayLike,
    *,
    method: str = "spearman",
    block_length: int,
    n_resamples: int = 1_000,
    confidence: float = 0.95,
    seed: int = 42,
    valid_mask: Sequence[bool] | None = None,
    segment_ids: Sequence[Any] | None = None,
) -> dict[str, Any]:
    """Paired moving-block interval for IC/rank IC."""
    return moving_block_bootstrap(
        _float_array(signal),
        _float_array(target),
        statistic=lambda x, y: safe_correlation(x, y, method=method),
        block_length=block_length,
        n_resamples=n_resamples,
        confidence=confidence,
        seed=seed,
        valid_mask=valid_mask,
        segment_ids=segment_ids,
    )


def fit_quantile_edges(signal: ArrayLike, requested_buckets: int = 10) -> np.ndarray:
    """Fit tie-preserving validation cutpoints, with open outer test buckets."""
    if requested_buckets < 1:
        raise ValueError("requested_buckets must be positive")
    values = _float_array(signal)
    values = values[np.isfinite(values)]
    if not len(values):
        raise ValueError("cannot fit quantiles without finite signals")
    if requested_buckets == 1 or np.ptp(values) == 0:
        return np.array([-np.inf, np.inf])
    cuts = np.quantile(values, np.arange(1, requested_buckets) / requested_buckets)
    cuts = np.unique(cuts[(cuts > np.min(values)) & (cuts < np.max(values))])
    return np.concatenate(([-np.inf], cuts.astype(float), [np.inf]))


def assign_quantile_buckets(signal: ArrayLike, edges: ArrayLike) -> np.ndarray:
    """Apply frozen edges; equal signals always receive the same 1-based bucket."""
    values = _float_array(signal)
    boundaries = _float_array(edges)
    if len(boundaries) < 2 or not np.all(np.diff(boundaries) > 0):
        raise ValueError("edges must be strictly increasing and contain at least two values")
    buckets = np.full(len(values), -1, dtype=int)
    finite = np.isfinite(values)
    buckets[finite] = np.searchsorted(boundaries[1:-1], values[finite], side="right") + 1
    return buckets


def quantile_analysis(
    signal: ArrayLike,
    future_return: ArrayLike,
    *,
    edges: ArrayLike | None = None,
    requested_buckets: int = 10,
    target_class: Sequence[int] | pd.Series | None = None,
    minimum_bucket_n: int = 1,
    include_sign_hit: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """OOS quantile table; pass validation-fitted ``edges`` for confirmatory use."""
    x, y = _float_array(signal), _float_array(future_return)
    if x.shape != y.shape:
        raise ValueError("signal and future_return must have equal length")
    fitted_here = edges is None
    boundaries = fit_quantile_edges(x, requested_buckets) if edges is None else _float_array(edges)
    buckets = assign_quantile_buckets(x, boundaries)
    classes = None if target_class is None else _float_array(target_class)
    if classes is not None and classes.shape != x.shape:
        raise ValueError("target_class length mismatch")

    rows: list[dict[str, Any]] = []
    actual_buckets = len(boundaries) - 1
    for bucket in range(1, actual_buckets + 1):
        in_bucket = (buckets == bucket) & np.isfinite(y)
        returns = y[in_bucket]
        signals = x[in_bucket]
        n = len(returns)
        directional = (signals != 0) & np.isfinite(signals)
        row: dict[str, Any] = {
            "bucket": bucket,
            "lower_edge": float(boundaries[bucket - 1]),
            "upper_edge": float(boundaries[bucket]),
            "n": int(n),
            "sufficient_n": bool(n >= minimum_bucket_n),
            "mean_return": float(np.mean(returns)) if n else float("nan"),
            "median_return": float(np.median(returns)) if n else float("nan"),
            "std_return": float(np.std(returns, ddof=1)) if n >= 2 else float("nan"),
            "positive_return_rate": float(np.mean(returns > 0)) if n else float("nan"),
            "negative_return_rate": float(np.mean(returns < 0)) if n else float("nan"),
            "zero_return_rate": float(np.mean(returns == 0)) if n else float("nan"),
            "sign_hit_rate": (
                float(np.mean(np.sign(signals[directional]) == np.sign(returns[directional])))
                if include_sign_hit and np.any(directional)
                else float("nan")
            ),
            "long_direction_accuracy": (
                float(np.mean(returns[signals > 0] > 0))
                if np.any(signals > 0)
                else float("nan")
            ),
            "short_direction_accuracy": (
                float(np.mean(returns[signals < 0] < 0))
                if np.any(signals < 0)
                else float("nan")
            ),
        }
        if classes is not None:
            class_values = classes[in_bucket]
            class_values = class_values[np.isfinite(class_values)]
            row.update(
                short_class_rate=float(np.mean(class_values == -1)) if len(class_values) else float("nan"),
                flat_class_rate=float(np.mean(class_values == 0)) if len(class_values) else float("nan"),
                long_class_rate=float(np.mean(class_values == 1)) if len(class_values) else float("nan"),
            )
        rows.append(row)

    table = pd.DataFrame(rows)
    means = table["mean_return"].to_numpy(dtype=float)
    finite_means = np.isfinite(means)
    adjacent = np.isfinite(means[:-1]) & np.isfinite(means[1:]) if len(means) > 1 else np.array([])
    metadata: dict[str, Any] = {
        "requested_bucket_count": int(requested_buckets),
        "actual_bucket_count": int(actual_buckets),
        "fitted_on_analysis_sample": bool(fitted_here),
        "finite_cutpoints": [float(v) for v in boundaries[1:-1]],
        "extreme_spread": (
            float(means[-1] - means[0])
            if len(means) >= 2 and np.isfinite(means[0]) and np.isfinite(means[-1])
            else float("nan")
        ),
        "adjacent_monotone_fraction": (
            float(np.mean(means[1:][adjacent] >= means[:-1][adjacent]))
            if np.any(adjacent)
            else float("nan")
        ),
        "bucket_rank_rho": (
            safe_correlation(np.arange(1, actual_buckets + 1)[finite_means], means[finite_means], "spearman")
            if np.sum(finite_means) >= 2
            else float("nan")
        ),
    }
    return table, metadata
