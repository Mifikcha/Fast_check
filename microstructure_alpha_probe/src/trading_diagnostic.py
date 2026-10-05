from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd


def prediction_actions(predictions: Sequence[float] | pd.Series, threshold: float) -> np.ndarray:
    """Map predictions to {-1, 0, +1}; equality and non-finite values are flat."""
    if threshold < 0 or not np.isfinite(threshold):
        raise ValueError("threshold must be finite and non-negative")
    values = pd.to_numeric(pd.Series(predictions), errors="coerce").to_numpy(dtype=float)
    actions = np.zeros(len(values), dtype=np.int8)
    actions[np.isfinite(values) & (values > threshold)] = 1
    actions[np.isfinite(values) & (values < -threshold)] = -1
    return actions


def _utc_timestamps(values: pd.Series) -> pd.Series:
    for value in values.dropna():
        if pd.Timestamp(value).tzinfo is None:
            raise ValueError("timestamps must be timezone-aware")
    parsed = pd.to_datetime(values, utc=True, errors="coerce")
    if parsed.isna().any():
        raise ValueError("timestamps contain missing or unparsable values")
    return parsed


def _positive(value: Any) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return bool(np.isfinite(number) and number > 0)


def _value(row: pd.Series, name: str) -> float:
    value = pd.to_numeric(pd.Series([row.get(name, np.nan)]), errors="coerce").iloc[0]
    return float(value) if np.isfinite(value) else float("nan")


def build_non_overlapping_trades(
    frame: pd.DataFrame,
    *,
    prediction_column: str,
    horizon_s: int,
    threshold: float,
    incremental_cost_bps: float = 0.0,
    timestamp_column: str = "timestamp",
    mid_column: str = "mid",
    bid_column: str = "best_bid",
    ask_column: str = "best_ask",
    book_valid_column: str | None = "book_valid",
    require_path_valid: bool = True,
) -> pd.DataFrame:
    """Create one-unit, fixed-horizon trades without overlapping positions.

    A future-invalid exit never changes the entry decision: after a valid entry,
    the position remains reserved through its scheduled exit and is counted as
    unpriceable. Future support columns are accepted only with an exact target
    end timestamp; otherwise quotes are looked up at the exact exit grid time.
    """
    if horizon_s <= 0:
        raise ValueError("horizon_s must be positive")
    if incremental_cost_bps < 0 or not np.isfinite(incremental_cost_bps):
        raise ValueError("incremental_cost_bps must be finite and non-negative")
    required = {timestamp_column, prediction_column, mid_column, bid_column, ask_column}
    missing = required - set(frame.columns)
    if missing:
        raise KeyError(f"Missing columns: {sorted(missing)}")

    data = frame.copy()
    data[timestamp_column] = _utc_timestamps(data[timestamp_column])
    data = data.sort_values(timestamp_column, kind="mergesort").reset_index(drop=True)
    if data[timestamp_column].duplicated().any():
        raise ValueError("timestamps must be unique for exact-horizon execution")
    actions = prediction_actions(data[prediction_column], threshold)
    data["_action"] = actions
    indexed = data.set_index(timestamp_column, drop=False)

    suffix = f"_{int(horizon_s)}s"
    target_end_column = f"target_end_time{suffix}"
    future_mid_column = f"future_mid{suffix}"
    future_bid_column = f"future_bid{suffix}"
    future_ask_column = f"future_ask{suffix}"
    target_valid_column = f"target_valid{suffix}"
    path_valid_column = f"target_path_valid{suffix}"
    has_support = {future_mid_column, future_bid_column, future_ask_column}.issubset(data.columns)

    period_start = data[timestamp_column].iloc[0] if len(data) else pd.NaT
    period_end = data[timestamp_column].iloc[-1] if len(data) else pd.NaT
    next_free_time: pd.Timestamp | None = None
    rejected_invalid_entry = 0
    skipped_overlap = 0
    rows: list[dict[str, Any]] = []
    horizon_delta = pd.Timedelta(seconds=int(horizon_s))
    cost_log = float(incremental_cost_bps) / 10_000.0

    for _, entry in data.iterrows():
        action = int(entry["_action"])
        if action == 0:
            continue
        entry_time = entry[timestamp_column]
        if next_free_time is not None and entry_time < next_free_time:
            skipped_overlap += 1
            continue
        if book_valid_column and book_valid_column in data and not bool(entry[book_valid_column]):
            rejected_invalid_entry += 1
            continue
        entry_mid, entry_bid, entry_ask = (
            _value(entry, mid_column),
            _value(entry, bid_column),
            _value(entry, ask_column),
        )
        relevant_entry = entry_ask if action == 1 else entry_bid
        if not (_positive(entry_mid) and _positive(relevant_entry)):
            rejected_invalid_entry += 1
            continue

        exit_time = entry_time + horizon_delta
        next_free_time = exit_time
        exit_row_present = False
        exact_target_end = True
        if has_support:
            if target_end_column in data and pd.notna(entry[target_end_column]):
                actual_end = pd.Timestamp(entry[target_end_column])
                if actual_end.tzinfo is None:
                    raise ValueError(f"{target_end_column} must be timezone-aware")
                exact_target_end = actual_end.tz_convert("UTC") == exit_time
            exit_mid = _value(entry, future_mid_column)
            exit_bid = _value(entry, future_bid_column)
            exit_ask = _value(entry, future_ask_column)
            exit_row_present = exact_target_end and any(_positive(v) for v in (exit_mid, exit_bid, exit_ask))
        elif exit_time in indexed.index:
            exit_row = indexed.loc[exit_time]
            if isinstance(exit_row, pd.DataFrame):
                raise ValueError("duplicate exact exit timestamp")
            exit_mid = _value(exit_row, mid_column)
            exit_bid = _value(exit_row, bid_column)
            exit_ask = _value(exit_row, ask_column)
            exit_row_present = True
        else:
            exit_mid = exit_bid = exit_ask = float("nan")

        endpoint_valid = exact_target_end
        if target_valid_column in data:
            endpoint_valid &= bool(entry[target_valid_column])
        path_valid = bool(entry[path_valid_column]) if path_valid_column in data else True
        allowed_exit = endpoint_valid and (path_valid or not require_path_valid)
        mid_priceable = allowed_exit and _positive(exit_mid)
        taker_exit = exit_bid if action == 1 else exit_ask
        taker_priceable = allowed_exit and _positive(taker_exit)
        gross_mid = action * math.log(exit_mid / entry_mid) if mid_priceable else float("nan")
        if taker_priceable:
            gross_taker = (
                math.log(exit_bid / entry_ask)
                if action == 1
                else math.log(entry_bid / exit_ask)
            )
        else:
            gross_taker = float("nan")

        prediction = _value(entry, prediction_column)
        rows.append(
            {
                "entry_time": entry_time,
                "exit_time": exit_time,
                "action": action,
                "side": "long" if action == 1 else "short",
                "prediction": prediction,
                "threshold": float(threshold),
                "horizon_s": int(horizon_s),
                "entry_mid": entry_mid,
                "entry_bid": entry_bid,
                "entry_ask": entry_ask,
                "exit_mid": exit_mid,
                "exit_bid": exit_bid,
                "exit_ask": exit_ask,
                "exact_target_end": bool(exact_target_end),
                "exit_row_present": bool(exit_row_present),
                "endpoint_valid": bool(endpoint_valid),
                "path_valid": bool(path_valid),
                "mid_priceable": bool(mid_priceable),
                "taker_priceable": bool(taker_priceable),
                "gross_mid_log": float(gross_mid),
                "gross_taker_log": float(gross_taker),
                "incremental_round_trip_cost_bps": float(incremental_cost_bps),
                "net_mid_log": float(gross_mid - cost_log) if np.isfinite(gross_mid) else float("nan"),
                "net_taker_log": (
                    float(gross_taker - cost_log) if np.isfinite(gross_taker) else float("nan")
                ),
                "holding_seconds": float(horizon_s),
            }
        )

    result = pd.DataFrame(rows)
    expected_columns = [
        "entry_time", "exit_time", "action", "side", "prediction", "threshold",
        "horizon_s", "entry_mid", "entry_bid", "entry_ask", "exit_mid", "exit_bid",
        "exit_ask", "exact_target_end", "exit_row_present", "endpoint_valid", "path_valid",
        "mid_priceable", "taker_priceable", "gross_mid_log", "gross_taker_log",
        "incremental_round_trip_cost_bps", "net_mid_log", "net_taker_log", "holding_seconds",
    ]
    if result.empty:
        result = pd.DataFrame(columns=expected_columns)
    result.attrs.update(
        period_start=period_start,
        period_end=period_end,
        candidate_signal_count=int(np.sum(actions != 0)),
        entry_count=int(len(result)),
        rejected_invalid_entry_count=int(rejected_invalid_entry),
        skipped_overlap_count=int(skipped_overlap),
        horizon_s=int(horizon_s),
        threshold=float(threshold),
        incremental_cost_bps=float(incremental_cost_bps),
    )
    return result


def equity_and_drawdown(
    trades: pd.DataFrame,
    *,
    return_column: str = "net_taker_log",
) -> pd.DataFrame:
    """Compounded unit-notional equity and drawdown ordered by exact exit time."""
    if return_column not in trades:
        raise KeyError(return_column)
    values = pd.to_numeric(trades[return_column], errors="coerce")
    keep = np.isfinite(values.to_numpy(dtype=float))
    columns = ["trade_no", "exit_time", "log_return", "equity", "equity_peak", "drawdown"]
    if not np.any(keep):
        return pd.DataFrame(columns=columns)
    completed = trades.loc[keep].copy()
    if "exit_time" in completed:
        completed = completed.sort_values("exit_time", kind="mergesort")
    log_returns = pd.to_numeric(completed[return_column], errors="coerce").to_numpy(dtype=float)
    equity = np.exp(np.cumsum(log_returns))
    peaks = np.maximum.accumulate(np.concatenate(([1.0], equity))) [1:]
    return pd.DataFrame(
        {
            "trade_no": np.arange(1, len(completed) + 1),
            "exit_time": completed.get("exit_time", pd.Series([pd.NaT] * len(completed))).to_numpy(),
            "log_return": log_returns,
            "equity": equity,
            "equity_peak": peaks,
            "drawdown": equity / peaks - 1.0,
        }
    )


def _exposure_fraction(trades: pd.DataFrame) -> float:
    start, end = trades.attrs.get("period_start"), trades.attrs.get("period_end")
    if pd.isna(start) or pd.isna(end) or end <= start or trades.empty:
        return float("nan") if pd.isna(start) or pd.isna(end) or end <= start else 0.0
    occupied = 0.0
    for _, row in trades.iterrows():
        left = max(pd.Timestamp(row["entry_time"]), pd.Timestamp(start))
        right = min(pd.Timestamp(row["exit_time"]), pd.Timestamp(end))
        occupied += max(0.0, (right - left).total_seconds())
    return float(min(1.0, occupied / (pd.Timestamp(end) - pd.Timestamp(start)).total_seconds()))


def trading_metrics(
    trades: pd.DataFrame,
    *,
    execution: str = "taker",
) -> dict[str, Any]:
    """Metrics for completed trades; undefined denominators remain NaN."""
    if execution not in {"mid", "taker"}:
        raise ValueError("execution must be 'mid' or 'taker'")
    gross_column, net_column = f"gross_{execution}_log", f"net_{execution}_log"
    if gross_column not in trades or net_column not in trades:
        raise KeyError(f"trades must contain {gross_column} and {net_column}")
    gross_all = pd.to_numeric(trades[gross_column], errors="coerce").to_numpy(dtype=float)
    net_all = pd.to_numeric(trades[net_column], errors="coerce").to_numpy(dtype=float)
    complete = np.isfinite(gross_all) & np.isfinite(net_all)
    gross, net = gross_all[complete], net_all[complete]
    n = len(net)
    positive, negative = net[net > 0], net[net < 0]
    wins, losses = float(np.sum(positive)), float(np.sum(negative))
    if not n:
        profit_factor, profit_factor_infinite = float("nan"), False
    elif losses == 0 and wins > 0:
        profit_factor, profit_factor_infinite = float("inf"), True
    elif wins == 0 and losses < 0:
        profit_factor, profit_factor_infinite = 0.0, False
    elif losses < 0:
        profit_factor, profit_factor_infinite = wins / abs(losses), False
    else:
        profit_factor, profit_factor_infinite = float("nan"), False
    sample_sd = float(np.std(net, ddof=1)) if n >= 2 else float("nan")
    sharpe_like = float(np.mean(net) / sample_sd) if np.isfinite(sample_sd) and sample_sd > 0 else float("nan")
    downside = np.minimum(net, 0.0)
    downside_deviation = float(np.sqrt(np.mean(downside**2))) if n else float("nan")
    sortino = (
        float(np.mean(net) / downside_deviation)
        if np.isfinite(downside_deviation) and downside_deviation > 0
        else float("nan")
    )
    curve = equity_and_drawdown(trades, return_column=net_column)
    actions = pd.to_numeric(trades.loc[complete, "action"], errors="coerce") if "action" in trades else pd.Series(dtype=float)
    holding = pd.to_numeric(trades.loc[complete, "holding_seconds"], errors="coerce") if "holding_seconds" in trades else pd.Series(dtype=float)
    entry_count = int(len(trades))
    return {
        "execution": execution,
        "entry_count": entry_count,
        "number_of_trades": int(n),
        "unpriceable_count": int(entry_count - n),
        "long_count": int(np.sum(actions == 1)),
        "short_count": int(np.sum(actions == -1)),
        "gross_pnl_log": float(np.sum(gross)) if n else 0.0,
        "net_pnl_log": float(np.sum(net)) if n else 0.0,
        "gross_pnl_bps": float(10_000 * np.sum(gross)) if n else 0.0,
        "net_pnl_bps": float(10_000 * np.sum(net)) if n else 0.0,
        "average_gross_return": float(np.mean(gross)) if n else float("nan"),
        "average_net_return": float(np.mean(net)) if n else float("nan"),
        "median_net_return": float(np.median(net)) if n else float("nan"),
        "return_std": sample_sd,
        "win_rate": float(np.mean(net > 0)) if n else float("nan"),
        "profit_factor": float(profit_factor),
        "profit_factor_infinite": bool(profit_factor_infinite),
        "sharpe_like_unannualized": sharpe_like,
        "sortino_unannualized": sortino,
        "max_drawdown": float(curve["drawdown"].min()) if len(curve) else float("nan"),
        "ending_equity": float(curve["equity"].iloc[-1]) if len(curve) else 1.0,
        "one_way_turnover": int(2 * entry_count),
        "priced_one_way_turnover": int(2 * n),
        "round_trips": entry_count,
        "average_holding_seconds": float(holding.mean()) if len(holding) else float("nan"),
        "exposure_fraction": _exposure_fraction(trades),
        "annualized": False,
    }


def cost_sensitivity(
    frame: pd.DataFrame,
    *,
    prediction_column: str,
    horizon_s: int,
    threshold: float,
    incremental_cost_bps: Sequence[float],
    **trade_kwargs: Any,
) -> pd.DataFrame:
    """Evaluate fixed actions/threshold under incremental round-trip costs."""
    rows: list[dict[str, Any]] = []
    for cost in incremental_cost_bps:
        trades = build_non_overlapping_trades(
            frame,
            prediction_column=prediction_column,
            horizon_s=horizon_s,
            threshold=threshold,
            incremental_cost_bps=float(cost),
            **trade_kwargs,
        )
        for execution in ("mid", "taker"):
            metrics = trading_metrics(trades, execution=execution)
            metrics.update(
                threshold=float(threshold),
                threshold_bps=float(threshold) * 10_000,
                incremental_round_trip_cost_bps=float(cost),
                horizon_s=int(horizon_s),
            )
            rows.append(metrics)
    return pd.DataFrame(rows)


def select_validation_threshold(
    validation: pd.DataFrame,
    *,
    prediction_column: str,
    horizon_s: int,
    threshold_grid_bps: Sequence[float],
    incremental_cost_bps: float,
    minimum_trades: int,
    objective: str = "average_net_return",
    stage: str = "validation",
    **trade_kwargs: Any,
) -> dict[str, Any]:
    """Select a numeric threshold on validation only and return the audit table."""
    if stage.lower() != "validation":
        raise ValueError("threshold selection is permitted only for stage='validation'")
    if minimum_trades < 1:
        raise ValueError("minimum_trades must be positive")
    if objective not in {"average_net_return", "net_pnl_log"}:
        raise ValueError("unsupported threshold objective")
    rows: list[dict[str, Any]] = []
    for threshold_bps in threshold_grid_bps:
        if threshold_bps < 0:
            raise ValueError("threshold_grid_bps cannot contain negative values")
        threshold = float(threshold_bps) / 10_000.0
        trades = build_non_overlapping_trades(
            validation,
            prediction_column=prediction_column,
            horizon_s=horizon_s,
            threshold=threshold,
            incremental_cost_bps=incremental_cost_bps,
            **trade_kwargs,
        )
        metrics = trading_metrics(trades, execution="taker")
        rows.append(
            {
                "threshold_bps": float(threshold_bps),
                "threshold": threshold,
                "eligible": bool(metrics["number_of_trades"] >= minimum_trades),
                **metrics,
            }
        )
    table = pd.DataFrame(rows)
    eligible = table.loc[table["eligible"] & np.isfinite(table[objective])].copy()
    if eligible.empty:
        return {
            "selected_threshold_bps": float("nan"),
            "selected_threshold": float("nan"),
            "objective": objective,
            "reason": "no_threshold_met_minimum_trades",
            "selection_table": table,
        }
    # Stable, declared ties: objective, total net PnL, then the smaller threshold.
    chosen = eligible.sort_values(
        [objective, "net_pnl_log", "threshold_bps"],
        ascending=[False, False, True],
        kind="mergesort",
    ).iloc[0]
    return {
        "selected_threshold_bps": float(chosen["threshold_bps"]),
        "selected_threshold": float(chosen["threshold"]),
        "objective": objective,
        "reason": None,
        "selection_table": table,
    }


def concentration_metrics(
    trades: pd.DataFrame,
    *,
    return_column: str = "net_taker_log",
    timestamp_column: str = "exit_time",
) -> dict[str, Any]:
    """Trade and calendar concentration, exposing unstable/negative denominators."""
    if return_column not in trades:
        raise KeyError(return_column)
    values = pd.to_numeric(trades[return_column], errors="coerce")
    keep = np.isfinite(values.to_numpy(dtype=float))
    clean = trades.loc[keep].copy()
    clean["_return"] = values.loc[keep].to_numpy(dtype=float)
    total = float(clean["_return"].sum()) if len(clean) else 0.0
    positive_total = float(clean.loc[clean["_return"] > 0, "_return"].sum()) if len(clean) else 0.0
    result: dict[str, Any] = {"n": int(len(clean)), "total_pnl": total, "positive_pnl": positive_total}
    ordered = np.sort(clean["_return"].to_numpy(dtype=float))[::-1] if len(clean) else np.array([])
    for fraction in (0.01, 0.05, 0.10):
        count = max(1, math.ceil(len(ordered) * fraction)) if len(ordered) else 0
        contribution = float(np.sum(ordered[:count])) if count else 0.0
        label = f"top_{int(fraction * 100)}pct"
        result[f"{label}_trade_count"] = int(count)
        result[f"{label}_pnl"] = contribution
        result[f"{label}_share_total_pnl"] = contribution / total if total != 0 else float("nan")
        result[f"{label}_share_positive_pnl"] = (
            contribution / positive_total if positive_total > 0 else float("nan")
        )

    if len(clean) and timestamp_column in clean:
        timestamps = pd.to_datetime(clean[timestamp_column], utc=True, errors="coerce")
        clean = clean.loc[timestamps.notna()].copy()
        clean["_timestamp"] = timestamps[timestamps.notna()]
        for label, period in (("hour", "h"), ("day", "D")):
            grouped = clean.groupby(clean["_timestamp"].dt.floor(period))["_return"].sum()
            if len(grouped):
                best_key = grouped.idxmax()
                best = float(grouped.loc[best_key])
                result[f"best_{label}"] = best_key.isoformat()
                result[f"best_{label}_pnl"] = best
                result[f"best_{label}_share_total_pnl"] = best / total if total != 0 else float("nan")
                result[f"pnl_without_best_{label}"] = total - best
                worst_key = grouped.idxmin()
                worst = float(grouped.loc[worst_key])
                result[f"worst_{label}"] = worst_key.isoformat()
                result[f"worst_{label}_pnl"] = worst
                result[f"worst_{label}_share_total_pnl"] = worst / total if total != 0 else float("nan")
            else:
                result[f"best_{label}"] = None
                result[f"best_{label}_pnl"] = float("nan")
                result[f"best_{label}_share_total_pnl"] = float("nan")
                result[f"pnl_without_best_{label}"] = float("nan")
                result[f"worst_{label}"] = None
                result[f"worst_{label}_pnl"] = float("nan")
                result[f"worst_{label}_share_total_pnl"] = float("nan")
    return result


def fit_regime_edges(values: Sequence[float] | pd.Series, quantiles: Sequence[float]) -> np.ndarray:
    """Fit unique development-only regime cutpoints with open outer intervals."""
    probabilities = np.asarray(quantiles, dtype=float)
    if np.any((probabilities <= 0) | (probabilities >= 1)) or np.any(np.diff(probabilities) <= 0):
        raise ValueError("quantiles must be strictly increasing inside (0,1)")
    data = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    data = data[np.isfinite(data)]
    if not len(data):
        raise ValueError("cannot fit regimes without finite values")
    cuts = np.unique(np.quantile(data, probabilities))
    cuts = cuts[(cuts > np.min(data)) & (cuts < np.max(data))]
    return np.concatenate(([-np.inf], cuts, [np.inf]))


def assign_regimes(
    values: Sequence[float] | pd.Series,
    edges: Sequence[float],
    labels: Sequence[str] = ("low", "medium", "high"),
) -> pd.Series:
    """Apply frozen regime edges without using the evaluation distribution."""
    data = pd.to_numeric(pd.Series(values), errors="coerce")
    boundaries = np.asarray(edges, dtype=float)
    if len(labels) != len(boundaries) - 1:
        raise ValueError("label count must equal len(edges)-1")
    assigned = pd.cut(data, bins=boundaries, labels=list(labels), include_lowest=True, right=True)
    return assigned


def regime_performance(
    frame: pd.DataFrame,
    *,
    regime_column: str,
    return_column: str,
) -> pd.DataFrame:
    """Simple regime return/PnL table, retaining categorical empty regimes."""
    if regime_column not in frame or return_column not in frame:
        raise KeyError(f"Missing {regime_column!r} or {return_column!r}")
    rows: list[dict[str, Any]] = []
    grouped = frame.groupby(regime_column, observed=False, dropna=False, sort=False)
    for regime, part in grouped:
        values = pd.to_numeric(part[return_column], errors="coerce").to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        rows.append(
            {
                "regime": None if pd.isna(regime) else str(regime),
                "n": int(len(values)),
                "pnl": float(np.sum(values)) if len(values) else 0.0,
                "mean_return": float(np.mean(values)) if len(values) else float("nan"),
                "median_return": float(np.median(values)) if len(values) else float("nan"),
                "win_rate": float(np.mean(values > 0)) if len(values) else float("nan"),
            }
        )
    return pd.DataFrame(rows)
