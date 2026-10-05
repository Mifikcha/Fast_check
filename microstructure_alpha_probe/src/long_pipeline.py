"""Incremental UTC-day processing and longitudinal Stage 2 tables."""

from __future__ import annotations

import json
import hashlib
import math
import uuid
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from scipy import stats

from .book import OrderBook
from .build_dataset import build_dataset
from .config import PROJECT_ROOT, project_path
from .io_utils import atomic_write_dataframe, atomic_write_json, atomic_write_text, dataset_fingerprint
from .long_analysis import (
    FrozenBaseline, _float, _intervals_from_mask, buffered_eligibility,
    load_frozen_baseline, nonoverlap_ic_by_phase,
)
from .long_collection import completed_utc_days, recording_sessions
from .statistics import assign_quantile_buckets, bootstrap_correlation, safe_correlation
from .targets import add_targets


TABLE_NAMES = [
    "daily_confirmatory_metrics.csv", "daily_model_metrics.csv", "daily_feature_ic.csv",
    "nonoverlap_ic_by_phase.csv", "latency_stress.csv", "horizon_comparison.csv",
    "obi_baselines.csv", "quantile_by_day.csv", "feature_group_by_day.csv",
    "regime_by_day.csv", "cost_sensitivity.csv", "selectivity_analysis.csv",
    "breakeven_costs.csv", "data_quality_by_day.csv", "invalid_intervals.csv",
]


def _day_frame(config: dict[str, Any], manifest: Path, day: str, *, partial: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    sessions = recording_sessions(manifest, project_path(config, "raw_dir"))
    start = pd.Timestamp(day, tz="UTC")
    end = start + pd.Timedelta(days=1)
    relevant = sessions.loc[(sessions["start"] < end) & (sessions["end"] >= start)] if not sessions.empty else sessions
    frames = []
    for row in relevant.itertuples():
        path = project_path(config, "processed_dir") / row.session_id / "research_table.parquet"
        if not path.exists():
            path = build_dataset(config, row.session_id)
        part = pd.read_parquet(path)
        part["timestamp"] = pd.to_datetime(part["timestamp"], utc=True)
        frames.append(part.loc[(part["timestamp"] >= start) & (part["timestamp"] < end)])
    if not frames:
        raise ValueError(f"no recorded data for UTC day {day}")
    observed = pd.concat(frames, ignore_index=True).sort_values("timestamp", kind="stable")
    overlap_times = pd.DatetimeIndex(observed.loc[observed["timestamp"].duplicated(keep=False), "timestamp"].unique())
    observed = observed.drop_duplicates("timestamp", keep="first")
    if partial:
        first, last = observed["timestamp"].min(), observed["timestamp"].max()
        grid = pd.date_range(first, last, freq="1s", tz="UTC")
    else:
        grid = pd.date_range(start, end - pd.Timedelta(seconds=1), freq="1s", tz="UTC")
    full = observed.set_index("timestamp").reindex(grid)
    full.index.name = "timestamp"
    full = full.reset_index()
    full["source_missing"] = full["session_id"].isna()
    full["source_overlap"] = full["timestamp"].isin(overlap_times)
    for col, default in (("book_valid", False), ("usable", False), ("gap_or_resync_interval", True)):
        full[col] = full[col].astype("boolean").fillna(default).astype(bool)
    full.loc[full["source_overlap"], ["book_valid", "usable"]] = False
    full.loc[full["source_overlap"], "gap_or_resync_interval"] = True
    source_invalid = full["source_missing"] | ~full["usable"] | full["gap_or_resync_interval"]
    return full, sessions.loc[relevant.index].copy() if not sessions.empty else sessions


def _prepare_day(frame: pd.DataFrame, frozen: FrozenBaseline) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    cfg = frozen.config["stage2"]
    horizons = [int(x) for x in cfg["horizons_s"]]
    source_missing = frame["source_missing"].to_numpy(bool)
    source_overlap = frame["source_overlap"].to_numpy(bool)
    invalid_book = ~frame["book_valid"].to_numpy(bool)
    resync = frame["gap_or_resync_interval"].to_numpy(bool) & ~source_missing
    stale = ~frame["usable"].to_numpy(bool) & ~source_missing & ~resync
    source = source_missing | source_overlap | invalid_book | resync | stale
    eligibility, merged = buffered_eligibility(
        frame["timestamp"], source,
        before_s=int(cfg["exclusion_before_s"]), after_s=int(cfg["exclusion_after_s"]), horizons_s=horizons,
    )
    intervals = []
    index = pd.DatetimeIndex(frame["timestamp"])
    for reason, mask in (("collector_downtime", source_missing), ("overlapping_sessions", source_overlap),
                         ("book_invalid", invalid_book & ~source_missing),
                         ("resynchronization", resync), ("stale_or_unusable", stale)):
        part = _intervals_from_mask(index, mask, reason)
        if not part.empty:
            part["buffered_start"] = part["start"] - pd.Timedelta(seconds=int(cfg["exclusion_before_s"]))
            part["buffered_end"] = part["end"] + pd.Timedelta(seconds=int(cfg["exclusion_after_s"]))
            intervals.append(part)
    interval_table = pd.concat(intervals, ignore_index=True) if intervals else merged.iloc[:0].copy()
    frame = frame.drop(columns=[col for col in frame if col.startswith(("target_", "future_"))], errors="ignore")
    frame = add_targets(frame, horizons_s=horizons, dead_zone_bps=2.0)
    for col in eligibility:
        if col != "timestamp":
            frame[col] = eligibility[col].to_numpy()
    for horizon in horizons:
        frame[f"eligible_{horizon}s"] &= frame[f"target_path_valid_{horizon}s"].fillna(False).astype(bool)
        frame[f"eligible_{horizon}s"] &= np.isfinite(frame[f"target_return_{horizon}s"])
    quality = {
        "raw_duration_s": len(frame), "invalid_duration_s": int(source.sum()),
        "excluded_buffered_duration_s": int(eligibility["buffered_invalid"].sum()),
        "final_usable_duration_s": int((~eligibility["buffered_invalid"]).sum()),
        "usable_pct": 100 * float((~eligibility["buffered_invalid"]).mean()),
        "collector_downtime_s": int(source_missing.sum()), "invalid_periods": len(merged),
    }
    return frame, interval_table, quality


def _metrics(signal: Any, target: Any) -> dict[str, Any]:
    x = pd.to_numeric(pd.Series(signal), errors="coerce").to_numpy(float)
    y = pd.to_numeric(pd.Series(target), errors="coerce").to_numpy(float)
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    return {"n": len(x), "pearson_ic": safe_correlation(x, y, "pearson"),
            "spearman_ic": safe_correlation(x, y, "spearman"),
            "rank_ic": safe_correlation(x, y, "spearman"),
            "sign_accuracy": float(np.mean(np.sign(x) == np.sign(y))) if len(x) else np.nan,
            "mean_prediction": float(np.mean(x)) if len(x) else np.nan,
            "prediction_std": float(np.std(x, ddof=1)) if len(x) > 1 else np.nan,
            "absolute_expected_move_bps": float(1e4 * np.mean(np.abs(y))) if len(y) else np.nan}


def _extremes(signal: Any, target: Any, edges: np.ndarray) -> dict[str, float]:
    x = np.asarray(signal, dtype=float)
    y = np.asarray(target, dtype=float)
    buckets = assign_quantile_buckets(x, edges)
    low = y[(buckets == 1) & np.isfinite(y)]
    high = y[(buckets == len(edges) - 1) & np.isfinite(y)]
    return {"bottom_mean_bps": float(1e4 * low.mean()) if len(low) else np.nan,
            "top_mean_bps": float(1e4 * high.mean()) if len(high) else np.nan,
            "extreme_bucket_spread_bps": float(1e4 * (high.mean() - low.mean())) if len(low) and len(high) else np.nan}


def _trades(frame: pd.DataFrame, horizon: int, *, threshold: float) -> pd.DataFrame:
    """Greedy non-overlap, exact target quotes, one unit per completed position."""
    predictions = frame["prediction"].to_numpy(float)
    eligible = frame[f"eligible_{horizon}s"].to_numpy(bool)
    candidates = np.flatnonzero(eligible & np.isfinite(predictions) & (np.abs(predictions) > threshold))
    if not len(candidates):
        return pd.DataFrame(columns=["entry_time", "action", "gross_mid_log", "gross_taker_log", "future_move_log"])
    timestamps = pd.DatetimeIndex(frame["timestamp"])
    chosen = []
    next_free = timestamps[0]
    for index in candidates:
        if timestamps[index] >= next_free:
            chosen.append(index)
            next_free = timestamps[index] + pd.Timedelta(seconds=horizon)
    rows = frame.iloc[chosen]
    action = np.sign(rows["prediction"].to_numpy(float))
    mid = rows["mid"].to_numpy(float)
    ask = rows["best_ask"].to_numpy(float)
    bid = rows["best_bid"].to_numpy(float)
    future_mid = rows[f"future_mid_{horizon}s"].to_numpy(float)
    future_bid = rows[f"future_bid_{horizon}s"].to_numpy(float)
    future_ask = rows[f"future_ask_{horizon}s"].to_numpy(float)
    gross_mid = action * np.log(future_mid / mid)
    gross_taker = np.where(action > 0, np.log(future_bid / ask), np.log(bid / future_ask))
    return pd.DataFrame({"entry_time": rows["timestamp"].to_numpy(), "action": action,
                         "gross_mid_log": gross_mid, "gross_taker_log": gross_taker,
                         "future_move_log": np.log(future_mid / mid)})


def _cost_rows(trades: pd.DataFrame, costs: list[float], *, day: str, horizon: int, model: str) -> list[dict[str, Any]]:
    mid = pd.to_numeric(trades.get("gross_mid_log", pd.Series(dtype=float)), errors="coerce").to_numpy(float)
    taker = pd.to_numeric(trades.get("gross_taker_log", pd.Series(dtype=float)), errors="coerce").to_numpy(float)
    mid, taker = mid[np.isfinite(mid)], taker[np.isfinite(taker)]
    count = len(taker)
    result = []
    for cost in costs:
        net = taker - cost / 1e4
        result.append({"day": day, "model": model, "horizon_s": horizon, "cost_bps": cost,
                       "trades": count, "gross_mid_pnl_log": float(mid.sum()),
                       "spread_adjusted_pnl_log": float(taker.sum()),
                       "net_taker_pnl_log": float(net.sum()),
                       "gross_mid_edge_bps": float(mid.mean() * 1e4) if len(mid) else np.nan,
                       "executable_edge_bps": float(taker.mean() * 1e4) if count else np.nan,
                       "net_edge_bps": float(net.mean() * 1e4) if count else np.nan,
                       "win_rate": float(np.mean(net > 0)) if count else np.nan,
                       "breakeven_cost_bps": max(0.0, float(taker.mean() * 1e4)) if count else np.nan,
                       "positive_before_cost": bool(count and taker.mean() > 0)})
    return result


def _phase_mean(frame: pd.DataFrame, signal: str, target: str, horizon: int) -> float:
    phase = (pd.DatetimeIndex(frame["timestamp"]).asi8 // 1_000_000_000) % horizon
    values = [safe_correlation(frame.loc[phase == p, signal], frame.loc[phase == p, target], "spearman")
              for p in range(horizon)]
    finite = np.asarray(values, dtype=float)
    return float(np.mean(finite[np.isfinite(finite)])) if np.isfinite(finite).any() else np.nan


def _quantiles(frame: pd.DataFrame, signal: str, target: str, edges: np.ndarray, *, day: str) -> list[dict[str, Any]]:
    buckets = assign_quantile_buckets(frame[signal], edges)
    returns = frame[target].to_numpy(float)
    means = []
    for bucket in range(1, len(edges)):
        values = returns[(buckets == bucket) & np.isfinite(returns)]
        means.append(float(values.mean()) if len(values) else np.nan)
    finite = np.isfinite(means)
    rho = safe_correlation(np.arange(1, len(edges))[finite], np.asarray(means)[finite], "spearman") if sum(finite) >= 2 else np.nan
    adjacent = [means[i + 1] >= means[i] for i in range(len(means) - 1) if np.isfinite(means[i]) and np.isfinite(means[i + 1])]
    return [{"day": day, "signal": signal, "bucket": i + 1, "bucket_name": f"Q{i+1}",
             "n": int(np.sum((buckets == i + 1) & np.isfinite(returns))), "mean_return_bps": 1e4 * value,
             "q10_minus_q1_bps": 1e4 * (means[-1] - means[0]) if np.isfinite(means[0]) and np.isfinite(means[-1]) else np.nan,
             "bucket_rank_rho": rho, "monotonicity_score": float(np.mean(adjacent)) if adjacent else np.nan}
            for i, value in enumerate(means)]


def _daily_tables(frame: pd.DataFrame, frozen: FrozenBaseline, *, day: str, complete: bool,
                  quality: dict[str, Any]) -> dict[str, pd.DataFrame]:
    cfg = frozen.config["stage2"]
    horizons = [int(x) for x in cfg["horizons_s"]]
    costs = [float(x) for x in cfg["cost_scenarios_bps"]]
    valid = frame["eligible_30s"].to_numpy(bool)
    primary = frame.loc[valid]
    confirm = {"day": day, "complete_utc_day": complete, **_metrics(primary["prediction"], primary["target_return_30s"]),
               **_extremes(primary["prediction"], primary["target_return_30s"], frozen.quantile_edges)}
    tables: dict[str, list[dict[str, Any]]] = {name: [] for name in TABLE_NAMES}
    tables["daily_confirmatory_metrics.csv"].append(confirm)
    tables["data_quality_by_day.csv"].append({"day": day, "complete_utc_day": complete, **quality,
        "quality_status": "BAD" if quality["usable_pct"] < float(cfg["minimum_global_usable_pct"]) else "WARN" if quality["invalid_duration_s"] else "GOOD"})
    phases = nonoverlap_ic_by_phase(primary.rename(columns={"target_return_30s": "actual"}), horizon_s=30)
    tables["nonoverlap_ic_by_phase.csv"].extend(({"day": day, **row}) for row in phases.to_dict("records"))
    for horizon in horizons:
        subset = frame.loc[frame[f"eligible_{horizon}s"]]
        target = f"target_return_{horizon}s"
        signal_metrics = _metrics(subset["prediction"], subset[target])
        extremes = _extremes(subset["prediction"], subset[target], frozen.quantile_edges)
        tables["horizon_comparison.csv"].append({"day": day, "model": "frozen_ridge30_signal", "horizon_s": horizon, **signal_metrics, **extremes})
        tables["daily_model_metrics.csv"].append({"day": day, "model": "frozen_ridge30_signal", "horizon_s": horizon, **signal_metrics})
        trades = _trades(frame, horizon, threshold=float(frozen.config["threshold"]))
        cost_rows = _cost_rows(trades, costs, day=day, horizon=horizon, model="frozen_ridge30_signal")
        tables["cost_sensitivity.csv"].extend(cost_rows)
        tables["breakeven_costs.csv"].append({"day": day, "model": "frozen_ridge30_signal", "horizon_s": horizon,
                                                "trades": len(trades), "breakeven_cost_bps": cost_rows[0]["breakeven_cost_bps"],
                                                "positive_before_cost": cost_rows[0]["positive_before_cost"]})
        if horizon == 30:
            confirm.update({"gross_mid_pnl_log": cost_rows[0]["gross_mid_pnl_log"],
                            "spread_adjusted_pnl_log": cost_rows[0]["spread_adjusted_pnl_log"],
                            "number_of_trades": len(trades)})
            for row in cost_rows:
                confirm[f"net_pnl_{row['cost_bps']:g}bps_log"] = row["net_taker_pnl_log"]
        for percent, threshold in frozen.selectivity_thresholds.items():
            selected = _trades(frame, horizon, threshold=threshold)
            result = _cost_rows(selected, [0.0], day=day, horizon=horizon, model="frozen_ridge30_signal")[0]
            future = selected["future_move_log"].to_numpy(float) if len(selected) else np.array([])
            tables["selectivity_analysis.csv"].append({"day": day, "horizon_s": horizon,
                "tail_percent_each_side": percent, "frozen_absolute_threshold": threshold,
                "mean_future_move_bps": float(np.mean(np.abs(future)) * 1e4) if len(future) else np.nan,
                "total_pnl_log": result["spread_adjusted_pnl_log"], **result})
        for signal in frozen.obi_edges:
            metrics = _metrics(subset[signal], subset[target])
            ext = _extremes(subset[signal], subset[target], frozen.obi_edges[signal])
            buckets = assign_quantile_buckets(subset[signal], frozen.obi_edges[signal])
            bucket_means = [subset.loc[buckets == i, target].mean() for i in range(1, len(frozen.obi_edges[signal]))]
            finite = np.isfinite(bucket_means)
            monotone = safe_correlation(np.arange(1, len(bucket_means) + 1)[finite], np.asarray(bucket_means)[finite], "spearman") if sum(finite) >= 2 else np.nan
            tables["obi_baselines.csv"].append({"day": day, "signal": signal, "horizon_s": horizon,
                **metrics, **ext, "nonoverlap_spearman_mean": _phase_mean(subset, signal, target, horizon),
                "quantile_monotonicity": monotone})
        for feature in frozen.features:
            tables["daily_feature_ic.csv"].append({"day": day, "feature": feature, "horizon_s": horizon,
                "n": int(np.isfinite(subset[feature]).sum()),
                "pearson_ic": safe_correlation(subset[feature], subset[target], "pearson"),
                "spearman_ic": safe_correlation(subset[feature], subset[target], "spearman")})
    for name, (columns, model) in frozen.group_models.items():
        prediction = model.predict(primary[columns]) if len(primary) else np.array([])
        tables["feature_group_by_day.csv"].append({"day": day, "feature_group": name,
            **_metrics(prediction, primary["target_return_30s"])})
    for signal, edges in (("frozen_ridge30_signal", frozen.quantile_edges), *frozen.obi_edges.items()):
        source = "prediction" if signal == "frozen_ridge30_signal" else signal
        tables["quantile_by_day.csv"].extend(_quantiles(primary, source, "target_return_30s", edges, day=day))
    for regime_kind, feature, edges in (("volatility", "realized_vol_60s", frozen.regime_edges["volatility"]),
                                        ("spread", "spread_bps", frozen.regime_edges["spread"])):
        levels = np.searchsorted(edges, primary[feature].to_numpy(float), side="right")
        for level, label in enumerate(("low", "medium", "high")):
            segment = primary.iloc[np.flatnonzero(levels == level)]
            segment_trades = _trades(segment, 30, threshold=float(frozen.config["threshold"])) if len(segment) else pd.DataFrame()
            tables["regime_by_day.csv"].append({"day": day, "regime_type": regime_kind, "regime": label,
                **_metrics(segment["prediction"], segment["target_return_30s"]),
                "trades": len(segment_trades), "gross_taker_edge_bps": float(segment_trades["gross_taker_log"].mean() * 1e4) if len(segment_trades) else np.nan})
    return {name: pd.DataFrame(rows) for name, rows in tables.items()}


def _raw_book_ticks(config: dict[str, Any], sessions: pd.DataFrame, *, start: pd.Timestamp,
                    end: pd.Timestamp) -> pd.DataFrame:
    """Replay every raw event once; keep subsecond L1 quotes and OBI for latency tests."""
    rows: list[tuple[int, float, float, float, bool]] = []
    for session in sessions.itertuples():
        book = OrderBook(symbol=config["project"]["symbol"])
        connection = None
        parts = sorted((project_path(config, "raw_dir") / "orderbook" / session.session_id).glob("part-*.parquet"))
        for path in parts:
            data = pd.read_parquet(path, columns=["recv_utc_ns", "connection_id", "raw_json", "payload_hash", "parse_ok"])
            for record in data.itertuples(index=False):
                if connection is not None and connection != record.connection_id:
                    book.mark_transport_loss("connection_changed")
                connection = record.connection_id
                if not bool(record.parse_ok):
                    book.invalidate("raw_parse_failure")
                else:
                    try:
                        book.process(json.loads(record.raw_json), payload_hash=record.payload_hash)
                    except (TypeError, ValueError):
                        book.invalidate("raw_json_failure")
                if record.recv_utc_ns < start.value or record.recv_utc_ns > end.value:
                    continue
                bid, ask = book.best_bid, book.best_ask
                if book.valid and bid and ask:
                    bid_price, bid_size = float(bid[0]), float(bid[1])
                    ask_price, ask_size = float(ask[0]), float(ask[1])
                    obi = (bid_size - ask_size) / (bid_size + ask_size) if bid_size + ask_size > 0 else np.nan
                    rows.append((int(record.recv_utc_ns), bid_price, ask_price, obi, True))
                else:
                    rows.append((int(record.recv_utc_ns), np.nan, np.nan, np.nan, False))
    ticks = pd.DataFrame(rows, columns=["recv_utc_ns", "best_bid", "best_ask", "obi_1", "book_valid"])
    if ticks.empty:
        return pd.DataFrame(columns=["timestamp", "best_bid", "best_ask", "obi_1", "book_valid"])
    ticks = ticks.sort_values("recv_utc_ns", kind="stable").reset_index(drop=True)
    ticks["timestamp"] = pd.to_datetime(ticks["recv_utc_ns"], unit="ns", utc=True)
    return ticks


def _latency_rows(frame: pd.DataFrame, frozen: FrozenBaseline, ticks: pd.DataFrame,
                  *, day: str) -> pd.DataFrame:
    """Two precise receive-time tests: stale OBI and delayed Ridge execution."""
    fields = ["day", "mode", "signal", "latency_ms", "n", "pearson_ic", "rank_ic",
              "extreme_bucket_spread_bps", "trades", "gross_mid_pnl_log",
              "taker_pnl_log", "average_taker_edge_bps", "net_1bps_pnl_log", "net_5bps_pnl_log"]
    if ticks.empty:
        return pd.DataFrame(columns=fields)
    primary = frame.loc[frame["eligible_30s"]].copy()
    if primary.empty:
        return pd.DataFrame(columns=fields)
    times = pd.DatetimeIndex(primary["timestamp"]).asi8
    ticks = ticks.loc[ticks["book_valid"]].reset_index(drop=True)
    if ticks.empty:
        return pd.DataFrame(columns=fields)
    events = ticks["recv_utc_ns"].to_numpy(np.int64)
    bid = ticks["best_bid"].to_numpy(float)
    ask = ticks["best_ask"].to_numpy(float)
    obi = ticks["obi_1"].to_numpy(float)
    prediction = primary["prediction"].to_numpy(float)
    original_return = primary["target_return_30s"].to_numpy(float)
    exit_deadline = times + 30 * 1_000_000_000

    def lookup(deadlines: np.ndarray, *, after: bool, max_age_ns: int) -> np.ndarray:
        idx = np.searchsorted(events, deadlines, side="left" if after else "right") + (0 if after else -1)
        in_bounds = (idx >= 0) & (idx < len(events))
        safe = np.clip(idx, 0, len(events) - 1)
        distance = np.abs(events[safe] - deadlines)
        in_bounds &= distance <= max_age_ns
        return np.where(in_bounds, safe, -1)

    exits = lookup(exit_deadline, after=True, max_age_ns=250_000_000)
    nominal = lookup(times, after=True, max_age_ns=250_000_000)
    rows = []
    for latency in [int(x) for x in frozen.config["stage2"]["latency_ms"]]:
        stale = lookup(times - latency * 1_000_000, after=False, max_age_ns=2_000_000_000)
        delayed = lookup(times + latency * 1_000_000, after=True, max_age_ns=250_000_000)
        for mode, signal, entries, outcome in (
            ("stale_obi1", np.where(stale >= 0, obi[np.clip(stale, 0, len(obi)-1)], np.nan), nominal, original_return),
            ("delayed_ridge30", prediction, delayed,
             np.where((delayed >= 0) & (exits >= 0),
                      np.log((bid[np.clip(exits, 0, len(bid)-1)] + ask[np.clip(exits, 0, len(ask)-1)]) /
                             (bid[np.clip(delayed, 0, len(bid)-1)] + ask[np.clip(delayed, 0, len(ask)-1)])), np.nan)),
        ):
            good = (entries >= 0) & (exits >= 0) & np.isfinite(signal) & np.isfinite(outcome)
            edges = frozen.obi_edges["obi_1"] if mode == "stale_obi1" else frozen.quantile_edges
            ext = _extremes(signal[good], outcome[good], edges)
            candidate = np.flatnonzero(good)
            if mode == "stale_obi1":
                buckets = assign_quantile_buckets(signal, edges)
                candidate = candidate[(buckets[candidate] == 1) | (buckets[candidate] == len(edges)-1)]
                action = np.where(buckets[candidate] == 1, -1, 1)
            else:
                candidate = candidate[np.abs(signal[candidate]) > float(frozen.config["threshold"])]
                action = np.sign(signal[candidate]).astype(int)
            chosen, actions = [], []
            next_free = -np.inf
            for i, direction in zip(candidate, action):
                if times[i] >= next_free:
                    chosen.append(i)
                    actions.append(direction)
                    next_free = times[i] + 30_000_000_000
            idx = np.asarray(chosen, dtype=int)
            actions = np.asarray(actions, dtype=int)
            if len(idx):
                entry_idx, exit_idx = entries[idx], exits[idx]
                entry_mid = (bid[entry_idx] + ask[entry_idx]) / 2
                exit_mid = (bid[exit_idx] + ask[exit_idx]) / 2
                gross = actions * np.log(exit_mid / entry_mid)
                taker = np.where(actions > 0, np.log(bid[exit_idx] / ask[entry_idx]),
                                 np.log(bid[entry_idx] / ask[exit_idx]))
            else:
                gross = taker = np.array([])
            rows.append({"day": day, "mode": mode, "signal": "OBI L1" if mode == "stale_obi1" else "frozen Ridge@30s",
                         "latency_ms": latency, "n": int(good.sum()),
                         "pearson_ic": safe_correlation(signal[good], outcome[good], "pearson"),
                         "rank_ic": safe_correlation(signal[good], outcome[good], "spearman"),
                         "extreme_bucket_spread_bps": ext["extreme_bucket_spread_bps"],
                         "trades": len(idx), "gross_mid_pnl_log": float(gross.sum()),
                         "taker_pnl_log": float(taker.sum()),
                         "average_taker_edge_bps": float(taker.mean()*1e4) if len(taker) else np.nan,
                         "net_1bps_pnl_log": float(taker.sum() - len(taker)/1e4),
                         "net_5bps_pnl_log": float(taker.sum() - 5*len(taker)/1e4)})
    return pd.DataFrame(rows, columns=fields)


def _experiment_days(manifest: Path, *, include_partial: bool, raw_root: Path) -> list[tuple[str, bool]]:
    meta = json.loads(manifest.read_text(encoding="utf-8"))
    sessions = recording_sessions(manifest, raw_root)
    if sessions.empty or sessions["end"].isna().all():
        return []
    first = pd.Timestamp(meta["started_utc"])
    last = min(pd.Timestamp.now(tz="UTC"), sessions["end"].max())
    complete = set(completed_utc_days(first, last))
    if include_partial:
        days = pd.date_range(first.floor("D"), last.floor("D"), freq="D")
        return [(day.strftime("%Y-%m-%d"), day.strftime("%Y-%m-%d") in complete) for day in days]
    return [(day, True) for day in sorted(complete)]


def analyze_day(
    config: dict[str, Any], manifest: Path, frozen: FrozenBaseline, *, day: str,
    complete: bool, progress: Callable[[dict[str, Any]], None] | None = None,
) -> Path:
    root = project_path(config, "runs_dir") / f"long_{manifest.parent.name}" / "daily"
    if complete:
        output = root / day
        if (output / "daily_manifest.json").exists():
            return output
    else:
        output = root / f"{day}_partial_{uuid.uuid4().hex[:8]}"
    if output.exists():
        output = root / f"{day}_{'complete' if complete else 'partial'}_{uuid.uuid4().hex[:8]}"
    output.mkdir(parents=True, exist_ok=False)
    if progress:
        progress({"phase": "daily", "message": f"{day}: loading session parts and exact UTC grid"})
    frame, sessions = _day_frame(config, manifest, day, partial=not complete)
    frame, intervals, quality = _prepare_day(frame, frozen)
    frame["prediction"] = frozen.model.predict(frame[frozen.features])
    if progress:
        progress({"phase": "daily", "message": f"{day}: computing day, horizon, non-overlap and economic tables"})
    tables = _daily_tables(frame, frozen, day=day, complete=complete, quality=quality)
    tables["invalid_intervals.csv"] = intervals.assign(day=day)
    if progress:
        progress({"phase": "daily", "message": f"{day}: replaying raw book ticks for millisecond latency"})
    ticks = _raw_book_ticks(config, sessions, start=frame["timestamp"].min() - pd.Timedelta(seconds=5),
                            end=frame["timestamp"].max() + pd.Timedelta(seconds=901))
    tables["latency_stress.csv"] = _latency_rows(frame, frozen, ticks, day=day)
    for name, table in tables.items():
        atomic_write_dataframe(table, output / name)
    prediction_columns = ["timestamp", "prediction", "target_return_30s", "eligible_30s", "source_missing",
                          "source_overlap", "buffered_invalid"]
    atomic_write_dataframe(frame[prediction_columns], output / "predictions.parquet")
    atomic_write_json({"day": day, "complete_utc_day": complete, "session_ids": sessions["session_id"].tolist(),
                       "rows": len(frame), "quality": quality, "frozen_source_run": frozen.config["source_run"],
                       "latency_semantics": {
                           "clock": "local receive timestamps (the actually available information clock)",
                           "stale_obi1": "latest raw book state at or before t-delta; max age 2s; t to t+30s target",
                           "delayed_ridge30": "frozen t signal; first valid raw quote at or after t+delta, exit at first quote at or after t+30s; max wait 250ms",
                       }}, output / "daily_manifest.json")
    atomic_write_text(
        f"# UTC {day} — {'complete' if complete else 'partial smoke'} day\n\n"
        f"Usable after buffers: {quality['usable_pct']:.2f}% ({quality['final_usable_duration_s']:,} seconds). "
        f"Raw invalid: {quality['invalid_duration_s']:,} seconds; buffered exclusion: {quality['excluded_buffered_duration_s']:,} seconds.\n\n"
        f"Frozen Ridge@30s rank IC: {_float(tables['daily_confirmatory_metrics.csv'].iloc[0].get('rank_ic')):.4f}. "
        "Inspect the CSV tables and latency semantics in `daily_manifest.json`.\n",
        output / "daily_report.md",
    )
    return output


def run_daily(
    config: dict[str, Any], manifest: Path, *, include_partial: bool = False,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> list[Path]:
    frozen = load_frozen_baseline(PROJECT_ROOT / "configs" / "confirmatory_ridge30.yaml")
    meta = json.loads(manifest.read_text(encoding="utf-8"))
    if meta["frozen_config_sha256"] != hashlib.sha256(
        (PROJECT_ROOT / "configs" / "confirmatory_ridge30.yaml").read_bytes()).hexdigest():
        raise ValueError("frozen config changed since experiment creation")
    days = _experiment_days(manifest, include_partial=include_partial, raw_root=project_path(config, "raw_dir"))
    if not days:
        raise ValueError("no completed UTC days; use --include-partial only for a stopped smoke session")
    return [analyze_day(config, manifest, frozen, day=day, complete=complete, progress=progress)
            for day, complete in days]
