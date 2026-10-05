from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _bool_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(False, index=frame.index, dtype=bool)
    values = frame[column]
    if pd.api.types.is_bool_dtype(values.dtype):
        return values.fillna(False).astype(bool)
    normalized = values.astype("string").str.strip().str.lower()
    return normalized.isin({"1", "true", "yes", "y"})


def _receive_times(frame: pd.DataFrame) -> pd.Series:
    if "recv_utc_ns" in frame:
        return pd.to_datetime(pd.to_numeric(frame["recv_utc_ns"], errors="coerce"), unit="ns", utc=True)
    for column in ("recv_time", "local_receive_time", "timestamp", "ts"):
        if column in frame:
            return pd.to_datetime(frame[column], utc=True, errors="coerce")
    return pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns, UTC]")


def _envelopes(frame: pd.DataFrame) -> pd.DataFrame:
    """Trade parts may be one row/item; collapse only for message-level timing."""
    keys = [column for column in ("connection_id", "local_message_no") if column in frame]
    return frame.drop_duplicates(keys, keep="first") if keys else frame


def _distribution(values: Sequence[float] | pd.Series) -> dict[str, Any]:
    numeric = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    numeric = numeric[np.isfinite(numeric)]
    if not len(numeric):
        return {
            "count": 0,
            "min": None,
            "p1": None,
            "p5": None,
            "median": None,
            "p90": None,
            "p95": None,
            "p99": None,
            "max": None,
            "negative_count": 0,
            "negative_pct": None,
        }
    quantiles = np.quantile(numeric, [0.01, 0.05, 0.50, 0.90, 0.95, 0.99])
    negative = int(np.sum(numeric < 0))
    return {
        "count": int(len(numeric)),
        "min": float(np.min(numeric)),
        "p1": float(quantiles[0]),
        "p5": float(quantiles[1]),
        "median": float(quantiles[2]),
        "p90": float(quantiles[3]),
        "p95": float(quantiles[4]),
        "p99": float(quantiles[5]),
        "max": float(np.max(numeric)),
        "negative_count": negative,
        "negative_pct": float(100 * negative / len(numeric)),
    }


def _spacing_metrics(times: pd.Series, missing_interval_s: float) -> dict[str, Any]:
    valid = times.dropna().sort_values(kind="mergesort")
    spacing = valid.diff().dt.total_seconds().dropna().to_numpy(dtype=float)
    spacing = spacing[np.isfinite(spacing) & (spacing >= 0)]
    gaps = spacing[spacing > missing_interval_s]
    distribution = _distribution(spacing)
    return {
        "event_spacing_s": distribution,
        "average_event_rate_hz": (
            float((len(valid) - 1) / (valid.iloc[-1] - valid.iloc[0]).total_seconds())
            if len(valid) >= 2 and valid.iloc[-1] > valid.iloc[0]
            else None
        ),
        "missing_interval_count": int(len(gaps)),
        "missing_interval_total_s": float(np.sum(gaps)) if len(gaps) else 0.0,
        "longest_missing_interval_s": float(np.max(gaps)) if len(gaps) else 0.0,
    }


def stream_data_quality(
    frame: pd.DataFrame | None,
    *,
    stream: str,
    missing_interval_s: float = 2.0,
) -> dict[str, Any]:
    """Summarize one raw stream without hiding malformed rows or clock skew."""
    if missing_interval_s <= 0:
        raise ValueError("missing_interval_s must be positive")
    data = pd.DataFrame() if frame is None else frame.copy()
    messages = _envelopes(data)
    receive = _receive_times(messages)
    first, last = receive.min(), receive.max()
    result: dict[str, Any] = {
        "stream": stream,
        "row_count": int(len(data)),
        "message_count": int(len(messages)),
        "first_receive_time": first.isoformat() if pd.notna(first) else None,
        "last_receive_time": last.isoformat() if pd.notna(last) else None,
        "duration_s": float((last - first).total_seconds()) if pd.notna(first) and pd.notna(last) else None,
    }
    result.update(_spacing_metrics(receive, missing_interval_s))
    median_spacing = result["event_spacing_s"].get("median")
    result["median_event_frequency_hz"] = (
        1.0 / median_spacing if median_spacing is not None and median_spacing > 0 else None
    )

    type_column = "type" if "type" in data else "message_type" if "message_type" in data else None
    types = data[type_column].astype("string").str.lower() if type_column else pd.Series("", index=data.index)
    result["snapshot_count"] = int((types == "snapshot").sum())
    result["delta_count"] = int((types == "delta").sum())
    result["duplicate_count"] = int(_bool_series(data, "duplicate").sum())
    result["sequence_anomaly_count"] = int(
        data["sequence_anomaly"].notna().sum() if "sequence_anomaly" in data else 0
    )
    result["proven_sequence_gap_count"] = int(
        _bool_series(data, "sequence_gap").sum()
        + _bool_series(data, "proven_gap").sum()
    )
    parse_bad = ~_bool_series(data, "parse_ok") if "parse_ok" in data else pd.Series(False, index=data.index)
    error_bad = data["error"].notna() if "error" in data else pd.Series(False, index=data.index)
    result["malformed_count"] = int((parse_bad | error_bad).sum())

    if stream == "orderbook":
        if {"n_bid_updates", "n_ask_updates"}.issubset(data.columns):
            updates = pd.to_numeric(data["n_bid_updates"], errors="coerce").fillna(0) + pd.to_numeric(
                data["n_ask_updates"], errors="coerce"
            ).fillna(0)
            result["book_level_update_count"] = int(updates.sum())
        else:
            result["book_level_update_count"] = None
        result["book_message_count"] = int((types.isin(["snapshot", "delta"])).sum())
        if "state_action" in data:
            actions = data["state_action"].astype("string").str.lower()
            result["resynchronization_count"] = int(actions.str.contains("resync", na=False).sum())
        else:
            result["resynchronization_count"] = 0
    if stream == "trades":
        if "trade_id" in data:
            result["trade_count"] = int(data["trade_id"].notna().sum())
            result["duplicate_trade_id_count"] = int(
                data.loc[data["trade_id"].notna()].duplicated(
                    [column for column in ("symbol", "trade_id") if column in data]
                ).sum()
            )
        elif "trade_count" in messages:
            result["trade_count"] = int(pd.to_numeric(messages["trade_count"], errors="coerce").fillna(0).sum())
            result["duplicate_trade_id_count"] = None
        else:
            result["trade_count"] = int(len(data))
            result["duplicate_trade_id_count"] = None

    exchange_column = "exchange_ts_ms" if "exchange_ts_ms" in messages else None
    if exchange_column:
        exchange_ms = pd.to_numeric(messages[exchange_column], errors="coerce").to_numpy(dtype=float)
        receive_ms = receive.astype("int64", copy=False).to_numpy(dtype=float) / 1_000_000
        valid = receive.notna().to_numpy() & np.isfinite(exchange_ms)
        result["exchange_to_receive_latency_ms"] = _distribution(receive_ms[valid] - exchange_ms[valid])
    else:
        result["exchange_to_receive_latency_ms"] = _distribution([])

    secondary = None
    if stream == "orderbook" and "matching_ts_ms" in messages:
        secondary = "matching_ts_ms"
    elif stream == "trades" and "trade_ts_ms" in messages:
        secondary = "trade_ts_ms"
    if secondary:
        event_ms = pd.to_numeric(messages[secondary], errors="coerce").to_numpy(dtype=float)
        receive_ms = receive.astype("int64", copy=False).to_numpy(dtype=float) / 1_000_000
        valid = receive.notna().to_numpy() & np.isfinite(event_ms)
        result["event_to_receive_latency_ms"] = _distribution(receive_ms[valid] - event_ms[valid])
    else:
        result["event_to_receive_latency_ms"] = _distribution([])
    return _json_safe(result)


def _event_counts(events: pd.DataFrame | None) -> dict[str, int]:
    if events is None or events.empty or "event_type" not in events:
        return {}
    normalized = events["event_type"].astype("string").str.strip().str.lower()
    return {str(key): int(value) for key, value in normalized.value_counts(dropna=False).items()}


def raw_data_quality(
    *,
    orderbook: pd.DataFrame | None = None,
    trades: pd.DataFrame | None = None,
    aux: pd.DataFrame | None = None,
    events: pd.DataFrame | None = None,
    missing_interval_s: float = 2.0,
) -> dict[str, Any]:
    """Raw quality payload with per-stream and session-event counts."""
    if events is None and aux is not None and "record_kind" in aux:
        event_mask = aux["record_kind"].astype("string").str.lower().isin(
            ["event", "collector_event"]
        )
        events = aux.loc[event_mask]
        aux = aux.loc[~event_mask]
    streams = {
        "orderbook": stream_data_quality(orderbook, stream="orderbook", missing_interval_s=missing_interval_s),
        "trades": stream_data_quality(trades, stream="trades", missing_interval_s=missing_interval_s),
        "aux": stream_data_quality(aux, stream="aux", missing_interval_s=missing_interval_s),
    }
    event_counts = _event_counts(events)
    totals = {
        "message_count": int(sum(item["message_count"] for item in streams.values())),
        "trade_count": int(streams["trades"].get("trade_count") or 0),
        "book_message_count": int(streams["orderbook"].get("book_message_count") or 0),
        "book_level_update_count": streams["orderbook"].get("book_level_update_count"),
        "snapshot_count": int(streams["orderbook"]["snapshot_count"]),
        "delta_count": int(streams["orderbook"]["delta_count"]),
        "duplicate_count": int(sum(item["duplicate_count"] for item in streams.values())),
        "malformed_count": int(sum(item["malformed_count"] for item in streams.values())),
        "sequence_anomaly_count": int(streams["orderbook"]["sequence_anomaly_count"]),
        "proven_sequence_gap_count": int(streams["orderbook"]["proven_sequence_gap_count"]),
        "reconnect_count": int(
            max(
                event_counts.get("disconnected", 0),
                max(0, event_counts.get("connecting", 0) - 1),
            )
        ),
        "resynchronization_count": int(
            event_counts.get("resynchronized", 0)
            + event_counts.get("resync", 0)
            + streams["orderbook"].get("resynchronization_count", 0)
        ),
        "queue_drop_count": int(event_counts.get("queue_drop", 0)),
        "write_failure_count": int(event_counts.get("write_failure", 0)),
    }
    # A reported sequence-gap count is explicitly limited to provable breaks.
    totals["sequence_gap_count"] = totals["proven_sequence_gap_count"]
    totals["sequence_gap_definition"] = "provable transport/local/ordering continuity breaks; not u/seq +1 jumps"
    return _json_safe({"streams": streams, "event_counts": event_counts, "totals": totals})


def _runs_on_grid(times: pd.Series, mask: pd.Series, interval_s: float) -> dict[str, Any]:
    valid_times = pd.to_datetime(times, utc=True, errors="coerce")
    flags = mask.fillna(False).astype(bool).to_numpy()
    lengths: list[int] = []
    current = 0
    previous: pd.Timestamp | None = None
    for time, flag in zip(valid_times, flags):
        contiguous = previous is not None and abs((time - previous).total_seconds() - interval_s) < 1e-9
        if flag:
            current = current + 1 if current and contiguous else 1
        elif current:
            lengths.append(current)
            current = 0
        if not contiguous and current > 1:
            lengths.append(current - 1)
            current = 1 if flag else 0
        previous = time
    if current:
        lengths.append(current)
    return {
        "period_count": int(len(lengths)),
        "total_s": float(sum(lengths) * interval_s),
        "longest_s": float(max(lengths) * interval_s) if lengths else 0.0,
    }


def processed_data_quality(
    frame: pd.DataFrame | None,
    *,
    horizons_s: Sequence[int] = (10, 30, 60),
    sampling_interval_s: float = 1.0,
    stale_book_s: float = 2.0,
    timestamp_column: str | None = None,
) -> dict[str, Any]:
    """Quality of the canonical grid; percentages use expected, not retained rows."""
    if sampling_interval_s <= 0 or stale_book_s < 0:
        raise ValueError("invalid sampling/staleness configuration")
    data = pd.DataFrame() if frame is None else frame.copy()
    timestamp_column = timestamp_column or ("timestamp" if "timestamp" in data else "ts")
    if timestamp_column not in data:
        return {
            "row_count": int(len(data)),
            "expected_grid_rows": 0,
            "timestamp_error": f"missing {timestamp_column}",
            "book_usable_pct": None,
            "model_usable_pct": {},
        }
    times = pd.to_datetime(data[timestamp_column], utc=True, errors="coerce")
    valid_times = times.dropna()
    duplicate_count = int(valid_times.duplicated().sum())
    monotonic = bool(times.notna().all() and times.is_monotonic_increasing)
    if len(valid_times):
        first, last = valid_times.min(), valid_times.max()
        expected_rows = int(round((last - first).total_seconds() / sampling_interval_s)) + 1
        duration_s = float((last - first).total_seconds() + sampling_interval_s)
    else:
        first = last = pd.NaT
        expected_rows = 0
        duration_s = None
    spacing = valid_times.sort_values(kind="mergesort").diff().dt.total_seconds().dropna()
    physical_gaps = spacing[spacing > sampling_interval_s + 1e-9]
    missing_grid_rows = int(
        sum(max(0, round(value / sampling_interval_s) - 1) for value in physical_gaps)
    )
    denominator = expected_rows
    usable_column = "book_valid" if "book_valid" in data else "usable" if "usable" in data else None
    book_usable = int(_bool_series(data, usable_column).sum()) if usable_column else 0
    model_usable_pct: dict[str, Any] = {}
    model_usable_count: dict[str, int] = {}
    feature_valid = _bool_series(data, "usable") if "usable" in data else _bool_series(data, "book_valid")
    for horizon in horizons_s:
        column = f"target_valid_{int(horizon)}s"
        valid = feature_valid & _bool_series(data, column) if column in data else pd.Series(False, index=data.index)
        count = int(valid.sum())
        model_usable_count[str(int(horizon))] = count
        model_usable_pct[str(int(horizon))] = float(100 * count / denominator) if denominator else None

    stale = (
        pd.to_numeric(data["book_age_ms"], errors="coerce") > stale_book_s * 1_000
        if "book_age_ms" in data
        else pd.Series(False, index=data.index)
    )
    gap_flag = _bool_series(data, "gap_or_resync_interval")
    if not gap_flag.any() and "gap_in_lookback_60s" in data:
        gap_flag = _bool_series(data, "gap_in_lookback_60s")
    result: dict[str, Any] = {
        "row_count": int(len(data)),
        "expected_grid_rows": int(expected_rows),
        "missing_grid_row_count": int(missing_grid_rows),
        "duplicate_timestamp_count": duplicate_count,
        "timestamps_monotonic": monotonic,
        "first_timestamp": first.isoformat() if pd.notna(first) else None,
        "last_timestamp": last.isoformat() if pd.notna(last) else None,
        "duration_s": duration_s,
        "physical_gap_count": int(len(physical_gaps)),
        "longest_physical_gap_s": float(physical_gaps.max()) if len(physical_gaps) else 0.0,
        "book_usable_count": book_usable,
        "book_usable_pct": float(100 * book_usable / denominator) if denominator else None,
        "model_usable_count": model_usable_count,
        "model_usable_pct": model_usable_pct,
        "stale_book_periods": _runs_on_grid(times, stale, sampling_interval_s),
        "gap_or_resync_periods": _runs_on_grid(times, gap_flag, sampling_interval_s),
    }
    if {"book_recv_utc_ns", "book_exchange_ts_ms"}.issubset(data.columns):
        receive_ms = pd.to_numeric(data["book_recv_utc_ns"], errors="coerce") / 1_000_000
        exchange_ms = pd.to_numeric(data["book_exchange_ts_ms"], errors="coerce")
        result["book_exchange_to_receive_latency_ms"] = _distribution(receive_ms - exchange_ms)
    return _json_safe(result)


def assess_data_quality(
    raw: Mapping[str, Any],
    processed: Mapping[str, Any],
    *,
    thresholds: Mapping[str, float | int],
) -> tuple[str, list[str]]:
    """Return explicit GOOD/WARN/BAD plus machine-readable reason strings."""
    totals = raw.get("totals", {})
    reasons_bad: list[str] = []
    reasons_warn: list[str] = []
    message_count = int(totals.get("message_count") or 0)
    if message_count == 0:
        reasons_bad.append("no_raw_messages")
    malformed = int(totals.get("malformed_count") or 0)
    if malformed > int(thresholds.get("max_malformed_messages", 0)):
        reasons_bad.append(f"malformed_messages={malformed}")
    elif malformed:
        reasons_warn.append(f"malformed_messages={malformed}")
    if int(totals.get("queue_drop_count") or 0):
        reasons_bad.append(f"queue_drops={totals['queue_drop_count']}")
    if int(totals.get("write_failure_count") or 0):
        reasons_bad.append(f"write_failures={totals['write_failure_count']}")
    if int(totals.get("proven_sequence_gap_count") or 0):
        reasons_warn.append(f"proven_sequence_gaps={totals['proven_sequence_gap_count']}")
    if int(totals.get("reconnect_count") or 0):
        reasons_warn.append(f"reconnects={totals['reconnect_count']}")
    if int(totals.get("duplicate_count") or 0):
        reasons_warn.append(f"duplicates={totals['duplicate_count']}")

    longest = max(
        (float(stream.get("longest_missing_interval_s") or 0) for stream in raw.get("streams", {}).values()),
        default=0.0,
    )
    if longest > float(thresholds.get("max_longest_gap_s", math.inf)):
        reasons_bad.append(f"longest_missing_interval_s={longest:g}")
    elif longest > 0:
        reasons_warn.append(f"missing_interval_s={longest:g}")

    negative_counts, latency_counts = 0, 0
    for stream in raw.get("streams", {}).values():
        latency = stream.get("exchange_to_receive_latency_ms", {})
        negative_counts += int(latency.get("negative_count") or 0)
        latency_counts += int(latency.get("count") or 0)
    negative_pct = 100 * negative_counts / latency_counts if latency_counts else 0.0
    if negative_pct > float(thresholds.get("max_negative_latency_pct", math.inf)):
        reasons_bad.append(f"negative_latency_pct={negative_pct:.6g}")
    elif negative_counts:
        reasons_warn.append(f"negative_latency_count={negative_counts}")

    if processed.get("timestamp_error"):
        reasons_bad.append(str(processed["timestamp_error"]))
    if int(processed.get("duplicate_timestamp_count") or 0):
        reasons_bad.append(f"duplicate_grid_timestamps={processed['duplicate_timestamp_count']}")
    if processed.get("timestamps_monotonic") is False:
        reasons_bad.append("grid_timestamps_not_monotonic")
    if int(processed.get("missing_grid_row_count") or 0):
        reasons_bad.append(f"missing_grid_rows={processed['missing_grid_row_count']}")
    usable = processed.get("book_usable_pct")
    if usable is not None and usable < float(thresholds.get("min_book_usable_pct", 0)):
        reasons_bad.append(f"book_usable_pct={usable:.6g}")
    elif usable is not None and usable < 100:
        reasons_warn.append(f"book_usable_pct={usable:.6g}")
    stale_count = int(processed.get("stale_book_periods", {}).get("period_count") or 0)
    if stale_count:
        reasons_warn.append(f"stale_book_periods={stale_count}")

    if reasons_bad:
        return "BAD", reasons_bad + reasons_warn
    if reasons_warn:
        return "WARN", reasons_warn
    return "GOOD", []


def generate_data_quality_payload(
    *,
    orderbook: pd.DataFrame | None = None,
    trades: pd.DataFrame | None = None,
    aux: pd.DataFrame | None = None,
    events: pd.DataFrame | None = None,
    processed: pd.DataFrame | None = None,
    horizons_s: Sequence[int] = (10, 30, 60),
    missing_interval_s: float = 2.0,
    sampling_interval_s: float = 1.0,
    stale_book_s: float = 2.0,
    thresholds: Mapping[str, float | int] | None = None,
) -> dict[str, Any]:
    """Build the canonical JSON-ready raw + processed quality payload."""
    limits = {
        "min_book_usable_pct": 95.0,
        "max_longest_gap_s": 10.0,
        "max_malformed_messages": 0,
        "max_negative_latency_pct": 1.0,
        **dict(thresholds or {}),
    }
    raw = raw_data_quality(
        orderbook=orderbook,
        trades=trades,
        aux=aux,
        events=events,
        missing_interval_s=missing_interval_s,
    )
    processed_metrics = processed_data_quality(
        processed,
        horizons_s=horizons_s,
        sampling_interval_s=sampling_interval_s,
        stale_book_s=stale_book_s,
    )
    status, reasons = assess_data_quality(raw, processed_metrics, thresholds=limits)
    return _json_safe(
        {
            "schema_version": 1,
            "status": status,
            "reasons": reasons,
            "thresholds": limits,
            "raw": raw,
            "processed": processed_metrics,
        }
    )


def _flatten(prefix: str, value: Any, output: dict[str, Any]) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            _flatten(f"{prefix}.{key}" if prefix else str(key), item, output)
    elif not isinstance(value, (list, tuple)):
        output[prefix] = value


def data_quality_csv(payload: Mapping[str, Any]) -> pd.DataFrame:
    """Produce a compact, one-row-per-scope CSV payload from quality JSON."""
    rows: list[dict[str, Any]] = []
    headline = {"scope": "overall", "status": payload.get("status"), "reasons": " | ".join(payload.get("reasons", []))}
    _flatten("raw.total", payload.get("raw", {}).get("totals", {}), headline)
    rows.append(headline)
    for stream_name, metrics in payload.get("raw", {}).get("streams", {}).items():
        row: dict[str, Any] = {"scope": f"raw:{stream_name}", "status": payload.get("status")}
        _flatten("metric", metrics, row)
        rows.append(row)
    processed_row: dict[str, Any] = {"scope": "processed", "status": payload.get("status")}
    _flatten("metric", payload.get("processed", {}), processed_row)
    rows.append(processed_row)
    return pd.DataFrame(rows)


def _display(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def data_quality_markdown(payload: Mapping[str, Any]) -> str:
    """Human-readable quality report with gaps and negative latency foregrounded."""
    status = str(payload.get("status", "BAD"))
    reasons = list(payload.get("reasons", []))
    lines = ["# Data quality report", "", f"## Status: {status}", ""]
    if reasons:
        lines += ["Reasons:", ""] + [f"- {reason}" for reason in reasons] + [""]
    else:
        lines += ["No configured quality warning or failure was detected.", ""]

    totals = payload.get("raw", {}).get("totals", {})
    lines += [
        "## Raw totals",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for key in (
        "message_count", "trade_count", "book_message_count", "book_level_update_count",
        "snapshot_count", "delta_count", "reconnect_count", "sequence_gap_count",
        "sequence_anomaly_count", "resynchronization_count", "duplicate_count",
        "malformed_count", "queue_drop_count", "write_failure_count",
    ):
        lines.append(f"| `{key}` | {_display(totals.get(key))} |")
    lines += ["", f"Sequence-gap definition: {totals.get('sequence_gap_definition', 'N/A')}", ""]

    lines += [
        "## Stream timing and latency",
        "",
        "| Stream | Messages | Duration s | Missing intervals | Longest gap s | Median spacing s | p99 spacing s | Negative latency count | Negative latency % |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, metrics in payload.get("raw", {}).get("streams", {}).items():
        spacing = metrics.get("event_spacing_s", {})
        latency = metrics.get("exchange_to_receive_latency_ms", {})
        lines.append(
            f"| {name} | {_display(metrics.get('message_count'))} | {_display(metrics.get('duration_s'))} "
            f"| {_display(metrics.get('missing_interval_count'))} | {_display(metrics.get('longest_missing_interval_s'))} "
            f"| {_display(spacing.get('median'))} | {_display(spacing.get('p99'))} "
            f"| {_display(latency.get('negative_count'))} | {_display(latency.get('negative_pct'))} |"
        )

    processed = payload.get("processed", {})
    lines += [
        "",
        "## Processed grid",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for key in (
        "row_count", "expected_grid_rows", "missing_grid_row_count", "duplicate_timestamp_count",
        "timestamps_monotonic", "book_usable_count", "book_usable_pct", "physical_gap_count",
        "longest_physical_gap_s",
    ):
        lines.append(f"| `{key}` | {_display(processed.get(key))} |")
    for horizon, value in processed.get("model_usable_pct", {}).items():
        lines.append(f"| `model_usable_pct_{horizon}s` | {_display(value)} |")
    stale = processed.get("stale_book_periods", {})
    lines += [
        f"| `stale_book_period_count` | {_display(stale.get('period_count'))} |",
        f"| `stale_book_total_s` | {_display(stale.get('total_s'))} |",
        f"| `stale_book_longest_s` | {_display(stale.get('longest_s'))} |",
        "",
        "Negative latency values are shown rather than clipped; they usually indicate host clock skew.",
        "",
    ]
    return "\n".join(lines)


def data_quality_payloads(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return ready-to-write JSON text, CSV dataframe, and Markdown text."""
    safe = _json_safe(dict(payload))
    return {
        "json": json.dumps(safe, indent=2, ensure_ascii=False, allow_nan=False),
        "csv": data_quality_csv(safe),
        "markdown": data_quality_markdown(safe),
    }
