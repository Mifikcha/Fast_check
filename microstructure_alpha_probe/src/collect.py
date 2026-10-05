"""Lossless public Bybit V5 WebSocket collector.

The collector reads the wire directly (not through pybit's reconstructed
callbacks), stores exact JSON, and writes immutable same-directory atomic ZSTD
Parquet parts.  Connection, parser, and book-state transitions are also written
as auxiliary events and JSONL for auditability.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import platform
import random
import shutil
import sys
import time
import uuid
from collections import Counter, OrderedDict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

try:  # Package import and direct `python src/collect.py` are both supported.
    from .book import OrderBook
except ImportError:  # pragma: no cover - exercised only by direct script use
    from book import OrderBook


SCHEMA_VERSION = "1"
DEFAULT_WS_URL = "wss://stream.bybit.com/v5/public/linear"


class CollectorError(RuntimeError):
    """Base collector error."""


class ReconnectRequired(CollectorError):
    """The current transport cannot be trusted and must be replaced."""


class CollectorFatalError(CollectorError):
    """Collection cannot continue without risking silent data loss."""


@dataclass(frozen=True, slots=True)
class CollectorConfig:
    raw_root: Path = Path("data/raw")
    symbol: str = "BTCUSDT"
    depth: int = 50
    ws_url: str = DEFAULT_WS_URL
    heartbeat_s: float = 20.0
    pong_timeout_s: float = 10.0
    stale_timeout_s: float = 10.0
    subscription_timeout_s: float = 10.0
    connect_timeout_s: float = 20.0
    close_timeout_s: float = 5.0
    reconnect_initial_s: float = 1.0
    reconnect_max_s: float = 30.0
    part_rows: int = 10_000
    part_seconds: float = 60.0
    websocket_max_queue: int = 1_024
    websocket_max_size: int = 16 * 1024 * 1024
    identity_cache_size: int = 20_000
    trade_dedup_cache_size: int = 500_000
    require_contiguous_book_ids: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "raw_root", Path(self.raw_root))
        object.__setattr__(self, "symbol", self.symbol.strip().upper())
        if not self.symbol:
            raise ValueError("symbol cannot be empty")
        if self.depth != 50:
            raise ValueError("the pilot raw schema is fixed to Bybit depth 50")
        for name in (
            "heartbeat_s",
            "pong_timeout_s",
            "stale_timeout_s",
            "subscription_timeout_s",
            "connect_timeout_s",
            "close_timeout_s",
            "reconnect_initial_s",
            "reconnect_max_s",
            "part_seconds",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        for name in (
            "part_rows",
            "websocket_max_queue",
            "websocket_max_size",
            "identity_cache_size",
            "trade_dedup_cache_size",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")

    @property
    def topics(self) -> tuple[str, str, str]:
        return (
            f"orderbook.{self.depth}.{self.symbol}",
            f"publicTrade.{self.symbol}",
            f"tickers.{self.symbol}",
        )

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "CollectorConfig":
        """Accept a flat mapping or the common nested YAML sections."""
        exchange = values.get("exchange") or values.get("bybit") or {}
        collector = values.get("collector") or values.get("collection") or {}
        paths = values.get("paths") or values.get("data") or {}
        if not isinstance(exchange, Mapping):
            exchange = {}
        if not isinstance(collector, Mapping):
            collector = {}
        if not isinstance(paths, Mapping):
            paths = {}

        def first(default: Any, *keys: str) -> Any:
            for source in (values, collector, exchange, paths):
                for key in keys:
                    if key in source and source[key] is not None:
                        return source[key]
            return default

        return cls(
            raw_root=Path(first("data/raw", "raw_root", "raw_dir")),
            symbol=str(first("BTCUSDT", "symbol")),
            depth=int(first(50, "depth", "orderbook_depth")),
            ws_url=str(first(DEFAULT_WS_URL, "ws_url", "websocket_url")),
            heartbeat_s=float(first(20, "heartbeat_s", "ping_interval_s", "heartbeat_interval_s")),
            pong_timeout_s=float(first(10, "pong_timeout_s")),
            stale_timeout_s=float(first(10, "stale_timeout_s", "stale_connection_s")),
            subscription_timeout_s=float(first(10, "subscription_timeout_s")),
            connect_timeout_s=float(first(20, "connect_timeout_s")),
            close_timeout_s=float(first(5, "close_timeout_s")),
            reconnect_initial_s=float(first(1, "reconnect_initial_s")),
            reconnect_max_s=float(first(30, "reconnect_max_s")),
            part_rows=int(first(10_000, "part_rows", "orderbook_part_rows")),
            part_seconds=float(first(60, "part_seconds", "flush_interval_s")),
            websocket_max_queue=int(first(1_024, "websocket_max_queue")),
            websocket_max_size=int(
                first(16 * 1024 * 1024, "websocket_max_size")
            ),
            identity_cache_size=int(first(20_000, "identity_cache_size")),
            trade_dedup_cache_size=int(
                first(500_000, "trade_dedup_cache_size")
            ),
            require_contiguous_book_ids=bool(
                first(False, "require_contiguous_book_ids", "strict_consecutive_update_ids")
            ),
        )


def _utc_iso(ns: int | None = None) -> str:
    timestamp = (time.time_ns() if ns is None else ns) / 1_000_000_000
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _payload_hash(raw: str) -> str:
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _optional_int(value: Any) -> int | None:
    if value is None or value == "" or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return None


def _optional_str(value: Any) -> str | None:
    return None if value is None else str(value)


def _optional_bool(value: Any) -> bool | None:
    return value if isinstance(value, bool) else None


def _schema(kind: str):
    """Build the frozen Arrow schema lazily, keeping module import lightweight."""
    try:
        import pyarrow as pa
    except ImportError as exc:  # pragma: no cover - dependency failure path
        raise CollectorFatalError(
            "pyarrow is required to write raw Parquet parts"
        ) from exc

    common = [
        pa.field("schema_version", pa.string()),
        pa.field("session_id", pa.string()),
        pa.field("connection_id", pa.string()),
        pa.field("local_message_no", pa.int64()),
        pa.field("recv_utc_ns", pa.int64()),
        pa.field("recv_monotonic_ns", pa.int64()),
        pa.field("payload_hash", pa.string()),
        pa.field("raw_json", pa.string()),
        pa.field("parse_ok", pa.bool_()),
    ]
    fields = {
        "orderbook": [
            pa.field("topic", pa.string()),
            pa.field("type", pa.string()),
            pa.field("exchange_ts_ms", pa.int64()),
            pa.field("matching_ts_ms", pa.int64()),
            pa.field("symbol", pa.string()),
            pa.field("update_id", pa.int64()),
            pa.field("cross_seq", pa.int64()),
            pa.field("bids_json", pa.string()),
            pa.field("asks_json", pa.string()),
            pa.field("n_bid_updates", pa.int32()),
            pa.field("n_ask_updates", pa.int32()),
            pa.field("state_action", pa.string()),
            pa.field("book_valid_after", pa.bool_()),
            pa.field("duplicate", pa.bool_()),
            pa.field("sequence_anomaly", pa.string()),
            pa.field("error", pa.string()),
        ],
        "trades": [
            pa.field("message_id", pa.string()),
            pa.field("trade_index", pa.int32()),
            pa.field("exchange_ts_ms", pa.int64()),
            pa.field("trade_ts_ms", pa.int64()),
            pa.field("symbol", pa.string()),
            pa.field("taker_side", pa.string()),
            pa.field("price", pa.string()),
            pa.field("quantity", pa.string()),
            pa.field("tick_direction", pa.string()),
            pa.field("trade_id", pa.string()),
            pa.field("is_block_trade", pa.bool_()),
            pa.field("is_rpi_trade", pa.bool_()),
            pa.field("cross_seq", pa.int64()),
            pa.field("duplicate", pa.bool_()),
            pa.field("error", pa.string()),
        ],
        "aux": [
            pa.field("record_kind", pa.string()),
            pa.field("event_type", pa.string()),
            pa.field("topic", pa.string()),
            pa.field("type", pa.string()),
            pa.field("exchange_ts_ms", pa.int64()),
            pa.field("cross_seq", pa.int64()),
            pa.field("symbol", pa.string()),
            pa.field("present_fields_json", pa.string()),
            pa.field("bid1_price", pa.string()),
            pa.field("bid1_size", pa.string()),
            pa.field("ask1_price", pa.string()),
            pa.field("ask1_size", pa.string()),
            pa.field("last_price", pa.string()),
            pa.field("mark_price", pa.string()),
            pa.field("index_price", pa.string()),
            pa.field("open_interest", pa.string()),
            pa.field("open_interest_value", pa.string()),
            pa.field("funding_rate", pa.string()),
            pa.field("next_funding_time_ms", pa.int64()),
            pa.field("funding_interval_hour", pa.string()),
            pa.field("reconnect_index", pa.int32()),
            pa.field("detail", pa.string()),
            pa.field("error", pa.string()),
        ],
    }
    try:
        return pa.schema(common + fields[kind])
    except KeyError as exc:
        raise ValueError(f"unknown raw schema {kind!r}") from exc


class AtomicParquetPartWriter:
    """Buffer rows and publish each compressed part exactly once."""

    def __init__(
        self,
        directory: Path,
        kind: str,
        *,
        max_rows: int,
        max_seconds: float,
    ) -> None:
        self.directory = directory
        self.kind = kind
        self.max_rows = max_rows
        self.max_seconds = max_seconds
        self.directory.mkdir(parents=True, exist_ok=False)
        self.rows: list[dict[str, Any]] = []
        self.part_index = 0
        self.last_flush = time.monotonic()
        self.parts: list[str] = []
        self.rows_written = 0
        self.bytes_written = 0

    def append(self, row: dict[str, Any]) -> None:
        self.rows.append(row)
        if len(self.rows) >= self.max_rows or (
            time.monotonic() - self.last_flush >= self.max_seconds
        ):
            self.flush()

    def flush(self) -> Path | None:
        if not self.rows:
            self.last_flush = time.monotonic()
            return None
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError as exc:  # pragma: no cover - dependency failure path
            raise CollectorFatalError(
                "pyarrow is required to write raw Parquet parts"
            ) from exc

        first = self.rows[0]["local_message_no"]
        last = self.rows[-1]["local_message_no"]
        unique = uuid.uuid4().hex
        filename = f"part-{self.part_index:06d}-{first:012d}-{last:012d}-{unique}.parquet"
        target = self.directory / filename
        temporary = self.directory / f".{filename}.tmp-{uuid.uuid4().hex}"
        if target.exists():
            raise CollectorFatalError(f"refusing to replace raw part: {target}")
        try:
            table = pa.Table.from_pylist(self.rows, schema=_schema(self.kind))
            pq.write_table(
                table,
                temporary,
                compression="zstd",
                use_dictionary=True,
                write_statistics=True,
            )
            # Windows rejects fsync on a read-only CRT descriptor.
            with temporary.open("rb+") as handle:
                os.fsync(handle.fileno())
            # Names include a UUID and are checked above. On Windows rename also
            # refuses an existing target, so published raw data is never replaced.
            os.rename(temporary, target)
        except Exception as exc:
            temporary.unlink(missing_ok=True)
            if isinstance(exc, CollectorFatalError):
                raise
            raise CollectorFatalError(
                f"failed to publish immutable {self.kind} Parquet part: {exc}"
            ) from exc
        self.parts.append(filename)
        self.part_index += 1
        self.rows_written += len(self.rows)
        self.bytes_written += target.stat().st_size
        self.rows.clear()
        self.last_flush = time.monotonic()
        return target

    def close(self) -> None:
        self.flush()


class StructuredEventLog:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=False)
        self.path = path
        self._handle = path.open("x", encoding="utf-8", buffering=1)

    def write(self, event_type: str, **details: Any) -> None:
        record = {
            "event_utc": _utc_iso(),
            "event_utc_ns": time.time_ns(),
            "event_monotonic_ns": time.monotonic_ns(),
            "event_type": event_type,
            **details,
        }
        self._handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._handle.flush()

    def close(self) -> None:
        if not self._handle.closed:
            self._handle.flush()
            os.fsync(self._handle.fileno())
            self._handle.close()


def _atomic_json_new(path: Path, payload: Mapping[str, Any]) -> None:
    """Publish a new JSON artifact atomically without replacing an old one."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise CollectorFatalError(f"refusing to replace session artifact: {path}")
    temporary = path.parent / f".{path.name}.tmp-{uuid.uuid4().hex}"
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, default=str)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.rename(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


class CollectorSession:
    """One independently auditable capture session."""

    def __init__(
        self,
        config: CollectorConfig,
        *,
        session_id: str | None = None,
        random_seed: int | None = None,
    ) -> None:
        self.config = config
        self.session_id = session_id or (
            datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
            + "-"
            + uuid.uuid4().hex[:12]
        )
        if any(char in self.session_id for char in "\\/:"):
            raise ValueError("session_id cannot contain path separators or ':'")
        self.random = random.Random(random_seed)
        self.start_utc_ns = time.time_ns()
        self.start_monotonic_ns = time.monotonic_ns()
        self.stop_monotonic: float | None = None
        self.connection_id = ""
        self.reconnect_index = 0
        self.local_message_no = 0
        self.counters: Counter[str] = Counter()
        self.book = OrderBook(
            symbol=config.symbol,
            identity_cache_size=config.identity_cache_size,
            require_contiguous_ids=config.require_contiguous_book_ids,
        )
        self._force_reconnect_reason: str | None = None
        self._trade_seen: OrderedDict[tuple[str, str], str] = OrderedDict()
        self._closed = False
        self._progress_callback: Callable[[dict[str, Any]], None] | None = None
        self._progress_interval_s = 30.0
        self._next_progress_monotonic = 0.0
        self._connection_state = "starting"
        self._last_wire_monotonic: float | None = None
        self._last_wire_utc_ns: int | None = None
        self._last_valid_book_utc_ns: int | None = None
        self._disk_reserve_bytes = 0

        session_meta = config.raw_root / "sessions" / self.session_id
        orderbook_dir = config.raw_root / "orderbook" / self.session_id
        trades_dir = config.raw_root / "trades" / self.session_id
        aux_dir = config.raw_root / "aux" / self.session_id
        for directory in (session_meta, orderbook_dir, trades_dir, aux_dir):
            if directory.exists():
                raise FileExistsError(f"capture session already exists: {directory}")

        self.event_log = StructuredEventLog(session_meta / "events.jsonl")
        self.session_dir = session_meta
        self.writers = {
            "orderbook": AtomicParquetPartWriter(
                orderbook_dir,
                "orderbook",
                max_rows=config.part_rows,
                max_seconds=config.part_seconds,
            ),
            "trades": AtomicParquetPartWriter(
                trades_dir,
                "trades",
                max_rows=config.part_rows,
                max_seconds=config.part_seconds,
            ),
            "aux": AtomicParquetPartWriter(
                aux_dir,
                "aux",
                max_rows=config.part_rows,
                max_seconds=config.part_seconds,
            ),
        }
        _atomic_json_new(session_meta / "session_start.json", self._manifest("running"))

    def _manifest(self, status: str, error: str | None = None) -> dict[str, Any]:
        config = asdict(self.config)
        config["raw_root"] = str(config["raw_root"])
        return {
            "schema_version": SCHEMA_VERSION,
            "session_id": self.session_id,
            "status": status,
            "start_utc_ns": self.start_utc_ns,
            "start_utc": _utc_iso(self.start_utc_ns),
            "end_utc_ns": None if status == "running" else time.time_ns(),
            "topics": list(self.config.topics),
            "config": config,
            "host_clock": {
                "host": platform.node(),
                "platform": platform.platform(),
                "python": sys.version,
                "time_ns_at_start": self.start_utc_ns,
                "monotonic_ns_at_start": self.start_monotonic_ns,
                "local_timezone_names": list(time.tzname),
            },
            "counters": dict(sorted(self.counters.items())),
            "parts": {
                kind: list(writer.parts) for kind, writer in self.writers.items()
            },
            "book": {
                "valid": self.book.valid,
                "invalid_reason": self.book.invalid_reason,
                "last_update_id": self.book.last_update_id,
                "last_cross_seq": self.book.last_cross_seq,
                "resynchronizations": self.book.resynchronizations,
            },
            "error": error,
        }

    def _next_message_no(self) -> int:
        self.local_message_no += 1
        return self.local_message_no

    def progress_snapshot(self, status: str | None = None) -> dict[str, Any]:
        """Return a cheap, read-only progress snapshot for console/UI reporting."""
        now = time.monotonic()
        elapsed = max(0.0, now - self.start_monotonic_ns / 1_000_000_000)
        remaining = self._remaining_seconds()
        writers = self.writers.values()
        total = None if remaining is None else elapsed + remaining
        return {
            "phase": "collect",
            "status": status or self._connection_state,
            "session_id": self.session_id,
            "elapsed_s": elapsed,
            "remaining_s": remaining,
            "percent": None if not total else min(100.0, 100.0 * elapsed / total),
            "wire_messages": self.counters["wire_message_count"],
            "message_rate_hz": self.counters["wire_message_count"] / elapsed if elapsed else 0.0,
            "book_messages": self.counters["orderbook_message_count"],
            "trades": self.counters["trade_count"],
            "ticker_messages": self.counters["ticker_message_count"],
            "connections": self.counters["connection_count"],
            "reconnects": self.counters["reconnect_count"],
            "book_valid": self.book.valid,
            "published_parts": sum(len(writer.parts) for writer in writers),
            "published_rows": sum(writer.rows_written for writer in self.writers.values()),
            "buffered_rows": sum(len(writer.rows) for writer in self.writers.values()),
            "saved_bytes": sum(writer.bytes_written for writer in self.writers.values()),
            "last_message_age_s": None if self._last_wire_monotonic is None else max(0.0, now - self._last_wire_monotonic),
            "last_message_utc": None if self._last_wire_utc_ns is None else _utc_iso(self._last_wire_utc_ns),
            "last_valid_book_utc": None if self._last_valid_book_utc_ns is None else _utc_iso(self._last_valid_book_utc_ns),
        }

    def _emit_progress(self, *, force: bool = False, status: str | None = None) -> None:
        now = time.monotonic()
        if self._progress_callback is None or (not force and now < self._next_progress_monotonic):
            return
        self._next_progress_monotonic = now + self._progress_interval_s
        if self._disk_reserve_bytes and shutil.disk_usage(self.config.raw_root).free < self._disk_reserve_bytes:
            raise CollectorFatalError("free disk space fell below configured long-run reserve")
        try:
            self._progress_callback(self.progress_snapshot(status))
        except Exception:
            # Diagnostics must never interrupt or corrupt market-data capture.
            self.counters["progress_callback_error_count"] += 1
            self._progress_callback = None

    def _common(
        self,
        *,
        message_no: int,
        recv_utc_ns: int,
        recv_monotonic_ns: int,
        raw: str,
        payload_hash: str,
        parse_ok: bool,
    ) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "session_id": self.session_id,
            "connection_id": self.connection_id,
            "local_message_no": message_no,
            "recv_utc_ns": recv_utc_ns,
            "recv_monotonic_ns": recv_monotonic_ns,
            "payload_hash": payload_hash,
            "raw_json": raw,
            "parse_ok": parse_ok,
        }

    def _internal_event(
        self,
        event_type: str,
        *,
        detail: str | None = None,
        error: str | None = None,
    ) -> None:
        recv_utc_ns, recv_monotonic_ns = time.time_ns(), time.monotonic_ns()
        payload = {
            "record_kind": "collector_event",
            "event_type": event_type,
            "detail": detail,
            "error": error,
            "session_id": self.session_id,
            "connection_id": self.connection_id,
            "reconnect_index": self.reconnect_index,
        }
        raw = _json_bytes(payload).decode("utf-8")
        message_no = self._next_message_no()
        self.writers["aux"].append(
            {
                **self._common(
                    message_no=message_no,
                    recv_utc_ns=recv_utc_ns,
                    recv_monotonic_ns=recv_monotonic_ns,
                    raw=raw,
                    payload_hash=_payload_hash(raw),
                    parse_ok=True,
                ),
                **self._empty_aux(),
                "record_kind": "collector_event",
                "event_type": event_type,
                "reconnect_index": self.reconnect_index,
                "detail": detail,
                "error": error,
            }
        )
        self.event_log.write(
            event_type,
            session_id=self.session_id,
            connection_id=self.connection_id,
            reconnect_index=self.reconnect_index,
            detail=detail,
            error=error,
        )

    @staticmethod
    def _empty_aux() -> dict[str, Any]:
        return {
            "record_kind": None,
            "event_type": None,
            "topic": None,
            "type": None,
            "exchange_ts_ms": None,
            "cross_seq": None,
            "symbol": None,
            "present_fields_json": None,
            "bid1_price": None,
            "bid1_size": None,
            "ask1_price": None,
            "ask1_size": None,
            "last_price": None,
            "mark_price": None,
            "index_price": None,
            "open_interest": None,
            "open_interest_value": None,
            "funding_rate": None,
            "next_funding_time_ms": None,
            "funding_interval_hour": None,
            "reconnect_index": None,
            "detail": None,
            "error": None,
        }

    def report_local_loss(self, reason: str) -> None:
        """Public hook for any downstream bounded-queue or disk-loss detector."""
        self.counters["local_loss_count"] += 1
        self.book.invalidate(f"local_loss:{reason}")
        self._force_reconnect_reason = reason
        self._internal_event("local_loss", error=reason)

    def _flush_all(self) -> None:
        try:
            for writer in self.writers.values():
                writer.flush()
        except Exception as exc:
            self.book.invalidate(f"raw_write_failure:{exc}")
            raise CollectorFatalError(f"raw Parquet write failed: {exc}") from exc

    def close(self, status: str, error: str | None = None) -> None:
        if self._closed:
            return
        final_error = error
        try:
            self._internal_event("session_end", detail=status, error=error)
            self._flush_all()
        except Exception as exc:
            status = "failed"
            final_error = final_error or str(exc)
        try:
            _atomic_json_new(
                self.session_dir / "session_end.json",
                self._manifest(status, final_error),
            )
        finally:
            self.event_log.close()
            self._closed = True
        if status == "failed" and final_error and error is None:
            raise CollectorFatalError(final_error)

    def handle_wire(
        self,
        raw: str | bytes,
        *,
        recv_utc_ns: int | None = None,
        recv_monotonic_ns: int | None = None,
    ) -> str | None:
        """Persist and dispatch one exact WebSocket payload.

        Returns a control event (`pong`, `subscription_ok`) for the connection
        loop. Structural failures are persisted first and then request a fresh
        connection/snapshot.
        """
        recv_utc_ns = time.time_ns() if recv_utc_ns is None else recv_utc_ns
        recv_monotonic_ns = (
            time.monotonic_ns() if recv_monotonic_ns is None else recv_monotonic_ns
        )
        self._last_wire_monotonic = recv_monotonic_ns / 1_000_000_000
        self._last_wire_utc_ns = recv_utc_ns
        if isinstance(raw, bytes):
            try:
                raw_text = raw.decode("utf-8")
            except UnicodeDecodeError:
                raw_text = raw.decode("utf-8", errors="replace")
                self._record_malformed(
                    raw_text,
                    "binary_frame_not_utf8",
                    recv_utc_ns,
                    recv_monotonic_ns,
                )
                self.book.invalidate("parse_failure:binary_frame_not_utf8")
                raise ReconnectRequired("binary_frame_not_utf8")
        elif isinstance(raw, str):
            raw_text = raw
        else:
            raw_text = repr(raw)
            self._record_malformed(
                raw_text,
                "non_text_websocket_frame",
                recv_utc_ns,
                recv_monotonic_ns,
            )
            self.book.invalidate("parse_failure:non_text_websocket_frame")
            raise ReconnectRequired("non_text_websocket_frame")

        self.counters["wire_message_count"] += 1
        try:
            message = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            self._record_malformed(
                raw_text,
                f"invalid_json:{exc}",
                recv_utc_ns,
                recv_monotonic_ns,
            )
            self.book.invalidate("parse_failure:invalid_json")
            raise ReconnectRequired("invalid_json") from exc
        if not isinstance(message, Mapping):
            self._record_malformed(
                raw_text,
                "json_root_not_object",
                recv_utc_ns,
                recv_monotonic_ns,
            )
            self.book.invalidate("parse_failure:json_root_not_object")
            raise ReconnectRequired("json_root_not_object")

        payload_hash = _payload_hash(raw_text)
        if message.get("op") == "ping" or message.get("ret_msg") == "pong":
            self._record_protocol(
                message,
                raw_text,
                payload_hash,
                recv_utc_ns,
                recv_monotonic_ns,
                "pong",
            )
            self.counters["pong_count"] += 1
            return "pong"
        if message.get("op") == "subscribe":
            ok = message.get("success") is True
            self._record_protocol(
                message,
                raw_text,
                payload_hash,
                recv_utc_ns,
                recv_monotonic_ns,
                "subscription_ok" if ok else "subscription_failed",
            )
            if not ok:
                raise ReconnectRequired(
                    f"subscription rejected: {message.get('ret_msg')!r}"
                )
            return "subscription_ok"

        topic = message.get("topic")
        if topic == self.config.topics[0]:
            self._record_orderbook(
                message,
                raw_text,
                payload_hash,
                recv_utc_ns,
                recv_monotonic_ns,
            )
        elif topic == self.config.topics[1]:
            self._record_trades(
                message,
                raw_text,
                payload_hash,
                recv_utc_ns,
                recv_monotonic_ns,
            )
        elif topic == self.config.topics[2]:
            self._record_ticker(
                message,
                raw_text,
                payload_hash,
                recv_utc_ns,
                recv_monotonic_ns,
            )
        else:
            self._record_protocol(
                message,
                raw_text,
                payload_hash,
                recv_utc_ns,
                recv_monotonic_ns,
                "unknown_message",
                error=f"unexpected topic/op: {topic!r}",
            )
            self.counters["unknown_message_count"] += 1
        return None

    def _record_orderbook(
        self,
        message: Mapping[str, Any],
        raw: str,
        payload_hash: str,
        recv_utc_ns: int,
        recv_monotonic_ns: int,
    ) -> None:
        message_no = self._next_message_no()
        data = message.get("data")
        data_map = data if isinstance(data, Mapping) else {}
        bids, asks = data_map.get("b"), data_map.get("a")
        result = self.book.process(message, payload_hash=payload_hash)
        if result.book_valid_after:
            self._last_valid_book_utc_ns = recv_utc_ns
        parse_ok = not (result.error or "").startswith("malformed_book_message")
        row = {
            **self._common(
                message_no=message_no,
                recv_utc_ns=recv_utc_ns,
                recv_monotonic_ns=recv_monotonic_ns,
                raw=raw,
                payload_hash=payload_hash,
                parse_ok=parse_ok,
            ),
            "topic": _optional_str(message.get("topic")),
            "type": _optional_str(message.get("type")),
            "exchange_ts_ms": _optional_int(message.get("ts")),
            "matching_ts_ms": _optional_int(message.get("cts")),
            "symbol": _optional_str(data_map.get("s")),
            "update_id": _optional_int(data_map.get("u")),
            "cross_seq": _optional_int(data_map.get("seq")),
            "bids_json": json.dumps(bids, ensure_ascii=False, separators=(",", ":")),
            "asks_json": json.dumps(asks, ensure_ascii=False, separators=(",", ":")),
            "n_bid_updates": len(bids) if isinstance(bids, list) else 0,
            "n_ask_updates": len(asks) if isinstance(asks, list) else 0,
            "state_action": result.state_action,
            "book_valid_after": result.book_valid_after,
            "duplicate": result.duplicate,
            "sequence_anomaly": result.sequence_anomaly,
            "error": result.error,
        }
        self.writers["orderbook"].append(row)
        self.counters["orderbook_message_count"] += 1
        self.counters[f"orderbook_{message.get('type', 'unknown')}_count"] += 1
        if result.duplicate:
            self.counters["duplicate_count"] += 1
        if result.sequence_anomaly:
            self.counters["sequence_diagnostic_count"] += 1
        if result.state_action == "resnapshot":
            self.counters["resynchronization_count"] += 1
        if not parse_ok:
            self.counters["malformed_message_count"] += 1
        if result.state_action == "invalidated":
            self.counters["book_invalidation_count"] += 1
            self.writers["orderbook"].flush()
            raise ReconnectRequired(result.error or "book_invalidated")

    def _record_trades(
        self,
        message: Mapping[str, Any],
        raw: str,
        payload_hash: str,
        recv_utc_ns: int,
        recv_monotonic_ns: int,
    ) -> None:
        message_no = self._next_message_no()
        trades = message.get("data")
        if not isinstance(trades, list):
            self._append_bad_trade(
                message_no,
                message,
                raw,
                payload_hash,
                recv_utc_ns,
                recv_monotonic_ns,
                "trade data must be an array",
            )
            self.book.invalidate("parse_failure:trade_data")
            raise ReconnectRequired("malformed trade data")
        self.counters["trade_message_count"] += 1
        malformed = False
        for trade_index, trade in enumerate(trades):
            if not isinstance(trade, Mapping):
                malformed = True
                self._append_bad_trade(
                    message_no,
                    message,
                    raw,
                    payload_hash,
                    recv_utc_ns,
                    recv_monotonic_ns,
                    f"trade[{trade_index}] is not an object",
                    trade_index,
                )
                continue
            symbol = _optional_str(trade.get("s"))
            trade_id = _optional_str(trade.get("i"))
            fingerprint = hashlib.sha256(_json_bytes(trade)).hexdigest()
            duplicate, error = False, None
            if not trade_id or not symbol:
                malformed = True
                error = "missing trade id or symbol"
            else:
                key = (symbol, trade_id)
                prior = self._trade_seen.get(key)
                if prior is not None:
                    duplicate = True
                    error = None if prior == fingerprint else "conflicting_trade_id"
                else:
                    self._trade_seen[key] = fingerprint
                    self._trade_seen.move_to_end(key)
                    while len(self._trade_seen) > self.config.trade_dedup_cache_size:
                        self._trade_seen.popitem(last=False)
            self.writers["trades"].append(
                {
                    **self._common(
                        message_no=message_no,
                        recv_utc_ns=recv_utc_ns,
                        recv_monotonic_ns=recv_monotonic_ns,
                        raw=raw,
                        payload_hash=payload_hash,
                        parse_ok=error is None or duplicate,
                    ),
                    "message_id": _optional_str(message.get("id")),
                    "trade_index": trade_index,
                    "exchange_ts_ms": _optional_int(message.get("ts")),
                    "trade_ts_ms": _optional_int(trade.get("T")),
                    "symbol": symbol,
                    "taker_side": _optional_str(trade.get("S")),
                    "price": _optional_str(trade.get("p")),
                    "quantity": _optional_str(trade.get("v")),
                    "tick_direction": _optional_str(trade.get("L")),
                    "trade_id": trade_id,
                    "is_block_trade": _optional_bool(trade.get("BT")),
                    "is_rpi_trade": _optional_bool(trade.get("RPI")),
                    "cross_seq": _optional_int(trade.get("seq")),
                    "duplicate": duplicate,
                    "error": error,
                }
            )
            self.counters["trade_count"] += 1
            if duplicate:
                self.counters["duplicate_count"] += 1
            if error:
                self.counters["malformed_trade_count"] += 1
        if malformed:
            self.book.invalidate("parse_failure:trade_record")
            self.writers["trades"].flush()
            raise ReconnectRequired("malformed trade record")

    def _append_bad_trade(
        self,
        message_no: int,
        message: Mapping[str, Any],
        raw: str,
        payload_hash: str,
        recv_utc_ns: int,
        recv_monotonic_ns: int,
        error: str,
        trade_index: int = 0,
    ) -> None:
        self.writers["trades"].append(
            {
                **self._common(
                    message_no=message_no,
                    recv_utc_ns=recv_utc_ns,
                    recv_monotonic_ns=recv_monotonic_ns,
                    raw=raw,
                    payload_hash=payload_hash,
                    parse_ok=False,
                ),
                "message_id": _optional_str(message.get("id")),
                "trade_index": trade_index,
                "exchange_ts_ms": _optional_int(message.get("ts")),
                "trade_ts_ms": None,
                "symbol": None,
                "taker_side": None,
                "price": None,
                "quantity": None,
                "tick_direction": None,
                "trade_id": None,
                "is_block_trade": None,
                "is_rpi_trade": None,
                "cross_seq": None,
                "duplicate": False,
                "error": error,
            }
        )
        self.counters["malformed_message_count"] += 1

    def _record_ticker(
        self,
        message: Mapping[str, Any],
        raw: str,
        payload_hash: str,
        recv_utc_ns: int,
        recv_monotonic_ns: int,
    ) -> None:
        message_no = self._next_message_no()
        data = message.get("data")
        if not isinstance(data, Mapping):
            self._record_aux_row(
                message_no=message_no,
                message=message,
                raw=raw,
                payload_hash=payload_hash,
                recv_utc_ns=recv_utc_ns,
                recv_monotonic_ns=recv_monotonic_ns,
                record_kind="ticker",
                event_type="malformed_ticker",
                parse_ok=False,
                error="ticker data must be an object",
            )
            self.book.invalidate("parse_failure:ticker_data")
            raise ReconnectRequired("malformed ticker data")
        self._record_aux_row(
            message_no=message_no,
            message=message,
            raw=raw,
            payload_hash=payload_hash,
            recv_utc_ns=recv_utc_ns,
            recv_monotonic_ns=recv_monotonic_ns,
            record_kind="ticker",
            event_type=_optional_str(message.get("type")),
            parse_ok=True,
            data=data,
        )
        self.counters["ticker_message_count"] += 1

    def _record_protocol(
        self,
        message: Mapping[str, Any],
        raw: str,
        payload_hash: str,
        recv_utc_ns: int,
        recv_monotonic_ns: int,
        event_type: str,
        *,
        error: str | None = None,
    ) -> None:
        self._record_aux_row(
            message_no=self._next_message_no(),
            message=message,
            raw=raw,
            payload_hash=payload_hash,
            recv_utc_ns=recv_utc_ns,
            recv_monotonic_ns=recv_monotonic_ns,
            record_kind="protocol",
            event_type=event_type,
            parse_ok=error is None,
            error=error,
        )

    def _record_malformed(
        self,
        raw: str,
        error: str,
        recv_utc_ns: int,
        recv_monotonic_ns: int,
    ) -> None:
        self._record_aux_row(
            message_no=self._next_message_no(),
            message={},
            raw=raw,
            payload_hash=_payload_hash(raw),
            recv_utc_ns=recv_utc_ns,
            recv_monotonic_ns=recv_monotonic_ns,
            record_kind="malformed",
            event_type="parse_error",
            parse_ok=False,
            error=error,
        )
        self.counters["malformed_message_count"] += 1
        self.writers["aux"].flush()

    def _record_aux_row(
        self,
        *,
        message_no: int,
        message: Mapping[str, Any],
        raw: str,
        payload_hash: str,
        recv_utc_ns: int,
        recv_monotonic_ns: int,
        record_kind: str,
        event_type: str | None,
        parse_ok: bool,
        data: Mapping[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        data = data or {}
        row = {
            **self._common(
                message_no=message_no,
                recv_utc_ns=recv_utc_ns,
                recv_monotonic_ns=recv_monotonic_ns,
                raw=raw,
                payload_hash=payload_hash,
                parse_ok=parse_ok,
            ),
            **self._empty_aux(),
            "record_kind": record_kind,
            "event_type": event_type,
            "topic": _optional_str(message.get("topic")),
            "type": _optional_str(message.get("type")),
            "exchange_ts_ms": _optional_int(message.get("ts")),
            "cross_seq": _optional_int(message.get("cs")),
            "symbol": _optional_str(data.get("symbol")),
            "present_fields_json": json.dumps(
                sorted(str(key) for key in data), ensure_ascii=False
            ),
            "bid1_price": _optional_str(data.get("bid1Price")),
            "bid1_size": _optional_str(data.get("bid1Size")),
            "ask1_price": _optional_str(data.get("ask1Price")),
            "ask1_size": _optional_str(data.get("ask1Size")),
            "last_price": _optional_str(data.get("lastPrice")),
            "mark_price": _optional_str(data.get("markPrice")),
            "index_price": _optional_str(data.get("indexPrice")),
            "open_interest": _optional_str(data.get("openInterest")),
            "open_interest_value": _optional_str(data.get("openInterestValue")),
            "funding_rate": _optional_str(data.get("fundingRate")),
            "next_funding_time_ms": _optional_int(data.get("nextFundingTime")),
            "funding_interval_hour": _optional_str(data.get("fundingIntervalHour")),
            "reconnect_index": self.reconnect_index,
            "detail": _optional_str(message.get("ret_msg")),
            "error": error,
        }
        self.writers["aux"].append(row)
        self.counters["aux_record_count"] += 1

    async def run(
        self,
        *,
        duration_s: float | None = None,
        stop_event: Any | None = None,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
        progress_interval_s: float = 30.0,
        disk_reserve_bytes: int = 0,
    ) -> Path:
        """Collect until duration/stop, reconnecting only through fresh snapshots."""
        if duration_s is not None and duration_s <= 0:
            raise ValueError("duration_s must be positive or None")
        if progress_interval_s <= 0:
            raise ValueError("progress_interval_s must be positive")
        self._progress_callback = progress_callback
        self._progress_interval_s = progress_interval_s
        self._disk_reserve_bytes = disk_reserve_bytes
        self.stop_monotonic = (
            None if duration_s is None else time.monotonic() + duration_s
        )
        status, failure = "complete", None
        reconnect_delay = self.config.reconnect_initial_s
        self._internal_event("session_start", detail=",".join(self.config.topics))
        self._connection_state = "connecting"
        try:
            while not self._should_stop(stop_event):
                self.connection_id = uuid.uuid4().hex
                if self.counters["connection_attempt_count"]:
                    self.reconnect_index += 1
                    self.counters["reconnect_count"] += 1
                    self.book.mark_transport_loss("transport_reconnect")
                self.counters["connection_attempt_count"] += 1
                self._connection_state = "connecting"
                self._internal_event("connecting", detail=self.config.ws_url)
                self._emit_progress(force=True)
                try:
                    await self._run_connection(stop_event)
                    reconnect_delay = self.config.reconnect_initial_s
                except asyncio.CancelledError:
                    status = "cancelled"
                    raise
                except CollectorFatalError:
                    raise
                except Exception as exc:
                    if self._should_stop(stop_event):
                        break
                    self.counters["disconnect_count"] += 1
                    self._connection_state = "reconnecting"
                    self.book.mark_transport_loss(f"transport_loss:{exc}")
                    self._internal_event("disconnected", error=str(exc))
                    self._flush_all()
                    self._emit_progress(force=True)
                    remaining = self._remaining_seconds()
                    if remaining is not None and remaining <= 0:
                        break
                    sleep_for = min(
                        reconnect_delay * self.random.uniform(0.8, 1.2),
                        remaining if remaining is not None else reconnect_delay * 1.2,
                    )
                    self._internal_event("reconnect_backoff", detail=f"{sleep_for:.3f}s")
                    await asyncio.sleep(max(0.0, sleep_for))
                    reconnect_delay = min(
                        reconnect_delay * 2, self.config.reconnect_max_s
                    )
        except asyncio.CancelledError:
            failure = "collection cancelled"
            raise
        except Exception as exc:
            status, failure = "failed", str(exc)
            raise
        finally:
            self.close(status, failure)
            self._connection_state = status
            self._emit_progress(force=True, status=status)
        return self.session_dir

    def _should_stop(self, stop_event: Any | None) -> bool:
        if self.stop_monotonic is not None and time.monotonic() >= self.stop_monotonic:
            return True
        return bool(stop_event is not None and stop_event.is_set())

    def _remaining_seconds(self) -> float | None:
        if self.stop_monotonic is None:
            return None
        return max(0.0, self.stop_monotonic - time.monotonic())

    async def _run_connection(self, stop_event: Any | None) -> None:
        try:
            from websockets.asyncio.client import connect
        except ImportError:
            try:
                from websockets import connect
            except ImportError as exc:  # pragma: no cover - dependency failure path
                raise CollectorFatalError(
                    "websockets is required for live collection"
                ) from exc

        async with connect(
            self.config.ws_url,
            ping_interval=None,
            open_timeout=self.config.connect_timeout_s,
            close_timeout=self.config.close_timeout_s,
            max_queue=self.config.websocket_max_queue,
            max_size=self.config.websocket_max_size,
        ) as websocket:
            self.counters["connection_count"] += 1
            self._connection_state = "connected"
            self._internal_event("connected")
            self._emit_progress(force=True)
            subscribe = json.dumps(
                {
                    "req_id": f"probe-{self.session_id}-{self.reconnect_index}",
                    "op": "subscribe",
                    "args": list(self.config.topics),
                },
                separators=(",", ":"),
            )
            await websocket.send(subscribe)
            self._internal_event("subscribe_sent", detail=subscribe)
            now = time.monotonic()
            last_receive = now
            next_ping = now + self.config.heartbeat_s
            subscription_deadline = now + self.config.subscription_timeout_s
            subscription_ok = False
            ping_sent_at: float | None = None

            while not self._should_stop(stop_event):
                now = time.monotonic()
                deadlines = [next_ping, last_receive + self.config.stale_timeout_s]
                if not subscription_ok:
                    deadlines.append(subscription_deadline)
                if ping_sent_at is not None:
                    deadlines.append(ping_sent_at + self.config.pong_timeout_s)
                remaining = self._remaining_seconds()
                if remaining is not None:
                    deadlines.append(now + remaining)
                timeout = max(0.01, min(deadlines) - now)
                try:
                    raw = await asyncio.wait_for(websocket.recv(), timeout=timeout)
                except asyncio.TimeoutError:
                    raw = None
                if raw is not None:
                    # Timestamp at the callback boundary, before JSON parsing or disk IO.
                    recv_utc_ns = time.time_ns()
                    recv_monotonic_ns = time.monotonic_ns()
                    last_receive = recv_monotonic_ns / 1_000_000_000
                    control = self.handle_wire(
                        raw,
                        recv_utc_ns=recv_utc_ns,
                        recv_monotonic_ns=recv_monotonic_ns,
                    )
                    if control == "pong":
                        ping_sent_at = None
                    elif control == "subscription_ok":
                        subscription_ok = True
                        self._connection_state = "subscribed"
                        self._internal_event("subscription_acknowledged")
                        self._emit_progress(force=True, status="subscribed")

                now = time.monotonic()
                if self._force_reconnect_reason:
                    reason, self._force_reconnect_reason = (
                        self._force_reconnect_reason,
                        None,
                    )
                    raise ReconnectRequired(f"local loss: {reason}")
                if not subscription_ok and now >= subscription_deadline:
                    raise ReconnectRequired("subscription acknowledgement timeout")
                if now - last_receive >= self.config.stale_timeout_s:
                    self.counters["stale_connection_count"] += 1
                    self._internal_event(
                        "stale_connection",
                        detail=f"no wire data for {now - last_receive:.3f}s",
                    )
                    raise ReconnectRequired("stale connection")
                if ping_sent_at is not None and (
                    now - ping_sent_at >= self.config.pong_timeout_s
                ):
                    self.counters["pong_timeout_count"] += 1
                    self._internal_event("pong_timeout")
                    raise ReconnectRequired("application pong timeout")
                if now >= next_ping:
                    ping = json.dumps(
                        {
                            "req_id": f"ping-{time.time_ns()}",
                            "op": "ping",
                        },
                        separators=(",", ":"),
                    )
                    await websocket.send(ping)
                    self.counters["ping_count"] += 1
                    ping_sent_at = now
                    next_ping = now + self.config.heartbeat_s
                    self._internal_event("ping_sent", detail=ping)
                self._emit_progress()


async def collect_session(
    config: CollectorConfig | Mapping[str, Any],
    *,
    duration_s: float | None = None,
    stop_event: Any | None = None,
    session_id: str | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    progress_interval_s: float = 30.0,
    disk_reserve_bytes: int = 0,
) -> Path:
    resolved = (
        config if isinstance(config, CollectorConfig) else CollectorConfig.from_mapping(config)
    )
    return await CollectorSession(resolved, session_id=session_id).run(
        duration_s=duration_s, stop_event=stop_event,
        progress_callback=progress_callback, progress_interval_s=progress_interval_s,
        disk_reserve_bytes=disk_reserve_bytes,
    )


def collect_blocking(
    config: CollectorConfig | Mapping[str, Any],
    *,
    duration_s: float | None = None,
    session_id: str | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    progress_interval_s: float = 30.0,
    disk_reserve_bytes: int = 0,
) -> Path:
    return asyncio.run(
        collect_session(
            config, duration_s=duration_s, session_id=session_id,
            progress_callback=progress_callback, progress_interval_s=progress_interval_s,
            disk_reserve_bytes=disk_reserve_bytes,
        )
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect public Bybit microstructure data")
    parser.add_argument("--hours", type=float, help="Stop after this many hours; omit for Ctrl+C")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--ws-url", default=DEFAULT_WS_URL)
    parser.add_argument("--part-rows", type=int, default=10_000)
    parser.add_argument("--part-seconds", type=float, default=60.0)
    parser.add_argument("--stale-timeout", type=float, default=10.0)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = CollectorConfig(
        raw_root=args.raw_root,
        symbol=args.symbol,
        ws_url=args.ws_url,
        part_rows=args.part_rows,
        part_seconds=args.part_seconds,
        stale_timeout_s=args.stale_timeout,
    )
    try:
        path = collect_blocking(
            config,
            duration_s=None if args.hours is None else args.hours * 3600,
        )
    except KeyboardInterrupt:
        print("Collection interrupted; buffered parts were finalized.")
        return
    print(path)


if __name__ == "__main__":
    main()


__all__ = [
    "AtomicParquetPartWriter",
    "CollectorConfig",
    "CollectorError",
    "CollectorFatalError",
    "CollectorSession",
    "ReconnectRequired",
    "collect_blocking",
    "collect_session",
]
