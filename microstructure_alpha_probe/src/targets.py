"""Exact-timestamp future-price targets for a complete UTC grid."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np
import pandas as pd


DEFAULT_HORIZONS_S = (10, 30, 60)


def _utc(values: object, name: str) -> pd.Series:
    try:
        converted = pd.to_datetime(values, utc=True, errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain parseable UTC timestamps") from exc
    return pd.Series(converted, index=getattr(values, "index", None), name=name)


def _duration(value: object, name: str) -> pd.Timedelta:
    result = pd.Timedelta(seconds=float(value)) if isinstance(value, (int, float, np.number)) else pd.Timedelta(value)
    if result <= pd.Timedelta(0):
        raise ValueError(f"{name} must be positive")
    return result


def _label(value: object) -> str:
    seconds = _duration(value, "horizon").total_seconds()
    return f"{int(seconds)}s" if float(seconds).is_integer() else f"{seconds:g}s"


def _dead_zone_bps(value: float | Mapping[object, float], horizon: object) -> float:
    if not isinstance(value, Mapping):
        result = float(value)
    else:
        seconds = _duration(horizon, "horizon").total_seconds()
        candidates = (
            horizon,
            int(seconds) if float(seconds).is_integer() else seconds,
            str(int(seconds)) if float(seconds).is_integer() else str(seconds),
            _label(horizon),
        )
        try:
            result = float(next(value[key] for key in candidates if key in value))
        except StopIteration as exc:
            raise ValueError(f"no dead-zone value configured for {_label(horizon)}") from exc
    if not np.isfinite(result) or result < 0:
        raise ValueError("dead_zone_bps must be finite and non-negative")
    return result


def directional_class(returns: pd.Series, dead_zone: float) -> pd.Series:
    """Classify valid log returns; values exactly on the boundary are flat."""
    if not np.isfinite(dead_zone) or dead_zone < 0:
        raise ValueError("dead_zone must be finite and non-negative")
    numeric = pd.to_numeric(returns, errors="coerce")
    result = pd.Series(pd.NA, index=returns.index, dtype="Int8")
    finite = np.isfinite(numeric)
    result.loc[finite] = 0
    result.loc[finite & numeric.lt(-dead_zone)] = -1
    result.loc[finite & numeric.gt(dead_zone)] = 1
    return result


def _future_values(
    values: pd.Series, timestamps: pd.Series, horizon: pd.Timedelta
) -> np.ndarray:
    indexed = pd.Series(values.to_numpy(), index=pd.DatetimeIndex(timestamps))
    return indexed.reindex(pd.DatetimeIndex(timestamps) + horizon).to_numpy()


def _path_validity(
    timestamps: pd.Series,
    point_valid: pd.Series,
    horizon: pd.Timedelta,
    step: pd.Timedelta,
) -> np.ndarray:
    """Require every expected grid point in the inclusive [t, t+H] path."""
    ratio = horizon / step
    if not float(ratio).is_integer():
        # Exact target timestamps can still exist on unusual grids, but a path
        # cannot be proved complete at a non-integral number of grid steps.
        return np.zeros(len(timestamps), dtype=bool)
    steps = int(ratio)
    times = pd.DatetimeIndex(timestamps)
    valid = point_valid.fillna(False).astype(bool).to_numpy()
    invalid_prefix = np.concatenate(([0], np.cumsum(~valid)))
    positions = np.arange(len(times))
    future_positions = positions + steps
    in_bounds = future_positions < len(times)
    exact = np.zeros(len(times), dtype=bool)
    exact[in_bounds] = (
        times.take(future_positions[in_bounds]).asi8
        == (times[in_bounds] + horizon).asi8
    )
    path_valid = np.zeros(len(times), dtype=bool)
    candidates = in_bounds & exact
    starts = positions[candidates]
    ends = future_positions[candidates]
    path_valid[candidates] = (invalid_prefix[ends + 1] - invalid_prefix[starts]) == 0
    return path_valid


def add_targets(
    grid: pd.DataFrame,
    *,
    horizons_s: Iterable[object] = DEFAULT_HORIZONS_S,
    dead_zone_bps: float | Mapping[object, float] = 0.0,
    timestamp_col: str = "timestamp",
    sampling_frequency: object = "1s",
    mid_col: str = "mid",
    bid_col: str = "best_bid",
    ask_col: str = "best_ask",
    valid_col: str = "book_valid",
    path_gap_col: str = "gap_or_resync_interval",
    dead_zone_conversion: str = "linear",
) -> pd.DataFrame:
    """Append continuous and directional targets using exact ``t + H`` joins.

    Invalid rows are retained. ``target_valid`` checks current/future endpoints;
    ``target_path_valid`` additionally checks every intervening grid point.
    """
    required = {timestamp_col, mid_col, bid_col, ask_col, valid_col}
    missing = required.difference(grid.columns)
    if missing:
        raise ValueError(f"grid is missing columns: {sorted(missing)}")
    result = grid.copy()
    result[timestamp_col] = _utc(result[timestamp_col], timestamp_col)
    if result[timestamp_col].duplicated().any():
        raise ValueError("target construction requires unique grid timestamps")
    if not result[timestamp_col].is_monotonic_increasing:
        raise ValueError("target construction requires chronologically sorted rows")

    step = _duration(sampling_frequency, "sampling_frequency")
    current_mid = pd.to_numeric(result[mid_col], errors="coerce")
    current_bid = pd.to_numeric(result[bid_col], errors="coerce")
    current_ask = pd.to_numeric(result[ask_col], errors="coerce")
    current_valid = result[valid_col].fillna(False).astype(bool)
    point_valid = (
        current_valid
        & np.isfinite(current_mid)
        & np.isfinite(current_bid)
        & np.isfinite(current_ask)
        & current_mid.gt(0)
        & current_bid.gt(0)
        & current_ask.gt(0)
        & current_bid.lt(current_ask)
    )
    if path_gap_col in result:
        point_valid &= ~result[path_gap_col].fillna(True).astype(bool)

    horizons = tuple(horizons_s)
    if not horizons:
        raise ValueError("at least one horizon is required")
    for horizon_value in horizons:
        horizon = _duration(horizon_value, "horizon")
        label = _label(horizon_value)
        future_time = result[timestamp_col] + horizon
        future_mid = pd.to_numeric(
            pd.Series(_future_values(current_mid, result[timestamp_col], horizon), index=result.index),
            errors="coerce",
        )
        future_bid = pd.to_numeric(
            pd.Series(_future_values(current_bid, result[timestamp_col], horizon), index=result.index),
            errors="coerce",
        )
        future_ask = pd.to_numeric(
            pd.Series(_future_values(current_ask, result[timestamp_col], horizon), index=result.index),
            errors="coerce",
        )
        future_point_valid = pd.Series(
            _future_values(point_valid.astype(bool), result[timestamp_col], horizon),
            index=result.index, dtype="boolean",
        ).fillna(False).astype(bool)
        endpoint_valid = (
            point_valid
            & future_point_valid
            & np.isfinite(future_mid)
            & np.isfinite(future_bid)
            & np.isfinite(future_ask)
            & future_mid.gt(0)
            & future_bid.gt(0)
            & future_ask.gt(0)
            & future_bid.lt(future_ask)
        )
        target_return = pd.Series(np.nan, index=result.index, dtype=float)
        target_return.loc[endpoint_valid] = np.log(
            future_mid.loc[endpoint_valid] / current_mid.loc[endpoint_valid]
        )

        configured_bps = _dead_zone_bps(dead_zone_bps, horizon_value)
        if dead_zone_conversion == "linear":
            dead_zone = configured_bps / 10_000
        elif dead_zone_conversion == "log1p":
            dead_zone = float(np.log1p(configured_bps / 10_000))
        else:
            raise ValueError("dead_zone_conversion must be 'linear' or 'log1p'")

        result[f"target_end_time_{label}"] = future_time
        result[f"future_mid_{label}"] = future_mid
        result[f"future_bid_{label}"] = future_bid
        result[f"future_ask_{label}"] = future_ask
        result[f"target_return_{label}"] = target_return
        result[f"target_class_{label}"] = directional_class(target_return, dead_zone)
        result[f"target_valid_{label}"] = endpoint_valid.astype(bool)
        result[f"target_path_valid_{label}"] = (
            endpoint_valid.to_numpy()
            & _path_validity(result[timestamp_col], point_valid, horizon, step)
        )
    return result


__all__ = ["add_targets", "directional_class"]
