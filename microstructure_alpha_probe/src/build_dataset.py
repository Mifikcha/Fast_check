"""Replay immutable raw parts into the causal one-second research table."""

from __future__ import annotations

import json
import time
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd

from .book import OrderBook
from .config import project_path
from .features import build_feature_table
from .io_utils import atomic_write_dataframe, atomic_write_json, dataset_fingerprint, list_parquet_parts
from .targets import add_targets


def _parts(config: dict[str, Any], kind: str, session_id: str) -> list[Path]:
    paths = list_parquet_parts(project_path(config, "raw_dir") / kind / session_id)
    if kind == "orderbook" and not paths:
        raise FileNotFoundError(f"no order-book parts for session {session_id!r}")
    return paths


def _read_parts(paths: Iterable[Path], columns: list[str] | None = None) -> Iterable[pd.DataFrame]:
    for path in paths:
        yield pd.read_parquet(path, columns=columns)


def _utc_from_ns(value: int) -> pd.Timestamp:
    return pd.Timestamp(int(value), unit="ns", tz="UTC")


def _bin(timestamp: pd.Timestamp, interval_s: int) -> pd.Timestamp:
    return timestamp.ceil(pd.Timedelta(seconds=interval_s))


def _session_bounds(config: dict[str, Any], session_id: str, book_parts: list[Path]) -> tuple[pd.Timestamp, pd.Timestamp]:
    manifest_dir = project_path(config, "raw_dir") / "sessions" / session_id
    start_path, end_path = manifest_dir / "session_start.json", manifest_dir / "session_end.json"
    if start_path.exists() and end_path.exists():
        start = json.loads(start_path.read_text(encoding="utf-8")).get("start_utc_ns")
        end = json.loads(end_path.read_text(encoding="utf-8")).get("end_utc_ns")
        if start is not None and end is not None:
            return _utc_from_ns(start), _utc_from_ns(end)
    first = pd.read_parquet(book_parts[0], columns=["recv_utc_ns"])["recv_utc_ns"]
    last = pd.read_parquet(book_parts[-1], columns=["recv_utc_ns"])["recv_utc_ns"]
    return _utc_from_ns(first.min()), _utc_from_ns(last.max())


def replay_order_book(
    paths: Iterable[Path], *, symbol: str, interval_s: int = 1, depth: int = 20,
    require_contiguous_ids: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Replay by local receive order and retain only the last state per grid bin."""
    book = OrderBook(symbol=symbol, require_contiguous_ids=require_contiguous_ids)
    states: dict[pd.Timestamp, dict[str, Any]] = {}
    ofi_sum: defaultdict[pd.Timestamp, float] = defaultdict(float)
    gap_bins: set[pd.Timestamp] = set()
    counters: defaultdict[str, int] = defaultdict(int)
    connection_id: str | None = None

    columns = [
        "session_id", "connection_id", "local_message_no", "recv_utc_ns",
        "raw_json", "payload_hash", "parse_ok",
    ]
    for frame in _read_parts(paths, columns):
        frame = frame.sort_values(["recv_utc_ns", "local_message_no"], kind="stable")
        for row in frame.itertuples(index=False):
            counters["raw_messages"] += 1
            timestamp = _utc_from_ns(row.recv_utc_ns)
            bucket = _bin(timestamp, interval_s)
            if connection_id is not None and row.connection_id != connection_id:
                book.mark_transport_loss("connection_changed")
                gap_bins.add(bucket)
                counters["connection_changes"] += 1
            connection_id = row.connection_id

            if not row.parse_ok:
                result = book.invalidate("raw_parse_failure")
                gap_bins.add(bucket)
                counters["parse_failures"] += 1
            else:
                try:
                    message = json.loads(row.raw_json)
                except (TypeError, json.JSONDecodeError):
                    result = book.invalidate("raw_json_decode_failure")
                    gap_bins.add(bucket)
                    counters["parse_failures"] += 1
                else:
                    result = book.process(message, payload_hash=row.payload_hash)
            if not result.book_valid_after:
                gap_bins.add(bucket)
                counters["invalid_messages"] += 1
            if result.state_action in {"snapshot", "resnapshot"}:
                gap_bins.add(bucket)
                counters["snapshots"] += 1
            if result.sequence_anomaly:
                counters["sequence_diagnostics"] += 1
            if result.duplicate:
                counters["duplicates"] += 1
            if result.ofi is not None and result.book_valid_after:
                ofi_sum[bucket] += float(result.ofi)

            state: dict[str, Any] = {
                "timestamp": timestamp,
                "session_id": row.session_id,
                "connection_id": row.connection_id,
                "symbol": symbol,
                "book_valid": bool(book.valid),
                "book_recv_utc_ns": int(row.recv_utc_ns),
                "book_exchange_ts_ms": book.last_exchange_ts_ms,
                "matching_ts_ms": book.last_matching_ts_ms,
                "update_id": book.last_update_id,
                "cross_seq": book.last_cross_seq,
                "state_action": result.state_action,
            }
            bids, asks = book.top(depth) if book.valid else ([], [])
            for level in range(1, depth + 1):
                for side, levels in (("bid", bids), ("ask", asks)):
                    price, size = levels[level - 1] if len(levels) >= level else (np.nan, np.nan)
                    state[f"{side}_price_{level:02d}"] = price
                    state[f"{side}_size_{level:02d}"] = size
            states[bucket] = state

    for bucket in gap_bins:
        if bucket in states:
            states[bucket]["gap_or_resync_interval"] = True
    for bucket, state in states.items():
        state.setdefault("gap_or_resync_interval", not state["book_valid"])

    state_frame = pd.DataFrame(states.values()).sort_values("timestamp", kind="stable")
    ofi_frame = pd.DataFrame(
        {"timestamp": list(ofi_sum), "ofi_event": list(ofi_sum.values())}
    )
    return state_frame, ofi_frame, dict(sorted(counters.items()))


def aggregate_trades(
    paths: Iterable[Path], grid: pd.DataFrame, *, interval_s: int,
    include_block: bool = True, include_rpi: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Aggregate valid unique trades by local availability time."""
    rows: list[pd.DataFrame] = []
    counters: defaultdict[str, int] = defaultdict(int)
    columns = [
        "recv_utc_ns", "parse_ok", "duplicate", "error", "taker_side",
        "price", "quantity", "trade_id", "is_block_trade", "is_rpi_trade",
    ]
    for frame in _read_parts(paths, columns):
        counters["raw_trade_rows"] += len(frame)
        good = frame["parse_ok"].fillna(False) & ~frame["duplicate"].fillna(False)
        if not include_block:
            good &= ~frame["is_block_trade"].fillna(False)
        if not include_rpi:
            good &= ~frame["is_rpi_trade"].fillna(False)
        part = frame.loc[good, ["recv_utc_ns", "taker_side", "quantity"]].copy()
        part["quantity"] = pd.to_numeric(part["quantity"], errors="coerce")
        valid = part["quantity"].ge(0) & part["taker_side"].astype("string").str.lower().isin(["buy", "sell"])
        counters["invalid_trade_rows"] += int((~valid).sum())
        rows.append(part.loc[valid])
    trades = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["recv_utc_ns", "taker_side", "quantity"])
    if not trades.empty:
        trades["timestamp"] = pd.to_datetime(trades.pop("recv_utc_ns"), unit="ns", utc=True)

    # A healthy subscribed interval with no trades is observed zero, not missing.
    base = grid[["timestamp", "book_valid", "gap_or_resync_interval"]].copy()
    base["buy_volume"] = base["sell_volume"] = 0.0
    base["buy_count"] = base["sell_count"] = 0
    base["coverage"] = (
        base["book_valid"].fillna(False).astype(bool)
        & ~base["gap_or_resync_interval"].fillna(True).astype(bool)
    ).astype(float)
    if not trades.empty:
        trades["_bin"] = trades["timestamp"].dt.ceil(pd.Timedelta(seconds=interval_s))
        side = trades["taker_side"].astype("string").str.lower()
        trades["buy_volume"] = trades["quantity"].where(side.eq("buy"), 0.0)
        trades["sell_volume"] = trades["quantity"].where(side.eq("sell"), 0.0)
        trades["buy_count"] = side.eq("buy").astype(int)
        trades["sell_count"] = side.eq("sell").astype(int)
        grouped = trades.groupby("_bin")[["buy_volume", "sell_volume", "buy_count", "sell_count"]].sum()
        base = base.set_index("timestamp")
        common = base.index.intersection(grouped.index)
        base.loc[common, grouped.columns] = grouped.loc[common].to_numpy()
        base = base.reset_index()
    counters["used_trade_rows"] = len(trades)
    return base[["timestamp", "buy_volume", "sell_volume", "buy_count", "sell_count", "coverage"]], dict(counters)


def _auxiliary_grid(paths: Iterable[Path], grid: pd.DataFrame, *, interval_s: int) -> pd.DataFrame:
    fields = {
        "bid1_price": "ticker_bid1", "ask1_price": "ticker_ask1",
        "open_interest": "open_interest", "funding_rate": "funding_rate",
        "next_funding_time_ms": "next_funding_time_ms",
    }
    state: dict[str, float] = {output: np.nan for output in fields.values()}
    source_ns: dict[str, float] = {output: np.nan for output in fields.values()}
    sampled: dict[pd.Timestamp, dict[str, Any]] = {}
    columns = ["recv_utc_ns", "local_message_no", "record_kind", "parse_ok", *fields]
    for frame in _read_parts(paths, columns):
        frame = frame.sort_values(["recv_utc_ns", "local_message_no"], kind="stable")
        frame = frame[frame["record_kind"].eq("ticker") & frame["parse_ok"].fillna(False)]
        for row in frame.itertuples(index=False):
            recv_ns = int(row.recv_utc_ns)
            for raw, output in fields.items():
                value = getattr(row, raw)
                if pd.notna(value):
                    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
                    if pd.notna(numeric):
                        state[output], source_ns[output] = float(numeric), recv_ns
            timestamp = _utc_from_ns(recv_ns)
            bucket = _bin(timestamp, interval_s)
            sampled[bucket] = {
                "timestamp": timestamp,
                "ticker_recv_utc_ns": recv_ns,
                **state,
                **{f"{name}_source_utc_ns": value for name, value in source_ns.items()},
            }
    result = grid[["timestamp"]].copy()
    if not sampled:
        for output in fields.values():
            result[output] = np.nan
        result["ticker_age_ms"] = np.nan
        result["open_interest_delta_60s"] = np.nan
        result["time_to_funding_s"] = np.nan
        return result
    events = pd.DataFrame(sampled.values()).sort_values("timestamp", kind="stable")
    result = pd.merge_asof(result, events, on="timestamp", direction="backward")
    receive = pd.to_datetime(result["ticker_recv_utc_ns"], unit="ns", utc=True)
    result["ticker_age_ms"] = (result["timestamp"] - receive).dt.total_seconds() * 1_000
    indexed = result.set_index("timestamp")["open_interest"]
    past = indexed.reindex(indexed.index - pd.Timedelta(seconds=60)).to_numpy(float)
    result["open_interest_delta_60s"] = indexed.to_numpy(float) - past
    result["time_to_funding_s"] = result["next_funding_time_ms"] / 1_000 - result["timestamp"].astype("int64") / 1e9
    return result.drop(columns=[c for c in result if c.endswith("_source_utc_ns")], errors="ignore")


def build_dataset(
    config: dict[str, Any], session_id: str, *,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> Path:
    """Build an immutable processed dataset and return its Parquet path."""
    started = time.monotonic()

    def notify(step: int, message: str, **details: Any) -> None:
        if progress:
            progress({"phase": "build", "step": step, "total": 8, "message": message,
                      "elapsed_s": time.monotonic() - started, **details})

    notify(1, "Поиск неизменяемых raw Parquet parts")
    interval_s = int(config["sampling"]["interval_s"])
    symbol = config["project"]["symbol"]
    book_parts = _parts(config, "orderbook", session_id)
    trade_parts, aux_parts = _parts(config, "trades", session_id), _parts(config, "aux", session_id)
    start, end = _session_bounds(config, session_id, book_parts)
    notify(2, "Воспроизведение snapshot и delta стакана", orderbook_parts=len(book_parts))
    states, ofi_events, replay_counts = replay_order_book(
        book_parts,
        symbol=symbol,
        interval_s=interval_s,
        depth=20,
        require_contiguous_ids=bool(config["collection"].get("strict_consecutive_update_ids", False)),
    )
    frequency = f"{interval_s}s"
    # First pass establishes the exact grid and book validity used as flow coverage.
    notify(3, "Построение причинной секундной сетки стакана", book_states=len(states))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", pd.errors.PerformanceWarning)
        preliminary = build_feature_table(
            states, start=start, end=end, frequency=frequency,
            max_book_age_ms=float(config["sampling"]["max_book_age_ms"]),
            depths=config["features"]["depth_levels"], trade_windows_s=[1],
            ofi_windows_s=[1], return_lags_s=[1], volatility_windows_s=[1],
        )
    notify(4, "Агрегация публичных сделок", trade_parts=len(trade_parts))
    trade_seconds, trade_counts = aggregate_trades(
        trade_parts, preliminary, interval_s=interval_s,
        include_block=bool(config["collection"].get("include_block_trades", True)),
        include_rpi=bool(config["collection"].get("include_rpi_trades", True)),
    )
    ofi_base = preliminary[["timestamp", "book_valid", "gap_or_resync_interval"]].copy()
    ofi_base["ofi_event"] = 0.0
    if not ofi_events.empty:
        grouped = ofi_events.assign(_bin=ofi_events["timestamp"].dt.ceil(frequency)).groupby("_bin")["ofi_event"].sum()
        common = pd.DatetimeIndex(ofi_base["timestamp"]).intersection(grouped.index)
        mapping = grouped.reindex(pd.DatetimeIndex(ofi_base["timestamp"]))
        ofi_base.loc[mapping.notna().to_numpy(), "ofi_event"] = mapping.dropna().to_numpy()
    ofi_base["coverage"] = (
        ofi_base["book_valid"].fillna(False).astype(bool)
        & ~ofi_base["gap_or_resync_interval"].fillna(True).astype(bool)
    ).astype(float)
    ofi_base = ofi_base[["timestamp", "ofi_event", "coverage"]]

    notify(5, "Расчёт причинных микроструктурных признаков")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", pd.errors.PerformanceWarning)
        research = build_feature_table(
            states, trade_seconds, ofi_base, start=start, end=end, frequency=frequency,
            max_book_age_ms=float(config["sampling"]["max_book_age_ms"]),
            depths=config["features"]["depth_levels"],
            trade_windows_s=config["features"]["trade_windows_s"],
            ofi_windows_s=config["features"]["ofi_windows_s"],
            return_lags_s=config["features"]["return_lags_s"],
            volatility_windows_s=config["features"]["volatility_windows_s"],
            numeric_epsilon=float(config["sampling"]["tiny_spread"]),
        )
    notify(6, "Присоединение ticker, OI, funding и точных future targets", aux_parts=len(aux_parts))
    aux = _auxiliary_grid(aux_parts, research, interval_s=interval_s)
    research = research.merge(aux, on="timestamp", how="left", validate="one_to_one")
    research = add_targets(
        research, horizons_s=config["targets"]["horizons_s"],
        dead_zone_bps=float(config["targets"]["dead_zone_bps"]),
        sampling_frequency=frequency,
    )
    research["session_id"] = session_id
    research["symbol"] = symbol

    notify(7, "Запись processed Parquet и манифеста", rows=len(research), columns=len(research.columns))
    output_dir = project_path(config, "processed_dir") / session_id
    output_dir.mkdir(parents=True, exist_ok=False)
    output = output_dir / "research_table.parquet"
    atomic_write_dataframe(research, output)
    all_parts = [*book_parts, *trade_parts, *aux_parts]
    atomic_write_json(
        {
            "session_id": session_id, "symbol": symbol,
            "start": research["timestamp"].min(), "end": research["timestamp"].max(),
            "rows": len(research), "usable_rows": int(research["usable"].sum()),
            "replay": replay_counts, "trades": trade_counts,
            "raw_fingerprint": dataset_fingerprint(all_parts),
        },
        output_dir / "dataset_manifest.json",
    )
    notify(8, "Processed dataset готов", rows=len(research), output=str(output))
    return output


__all__ = ["aggregate_trades", "build_dataset", "replay_order_book"]
