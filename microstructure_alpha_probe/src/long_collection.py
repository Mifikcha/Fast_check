"""Restartable Stage 2 recording using the existing immutable session/part format."""

from __future__ import annotations

import json
import os
import shutil
import time
import uuid
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import pyarrow.parquet as pq

from .collect import CollectorConfig, collect_blocking
from .config import PROJECT_ROOT, project_path
from .io_utils import atomic_write_json, git_commit, sha256_file


UTC = timezone.utc
LONG_ROOT = PROJECT_ROOT / "data" / "long"


def completed_utc_days(start: pd.Timestamp, end: pd.Timestamp) -> list[str]:
    """Calendar days wholly inside the observed experiment window."""
    first = pd.Timestamp(start).tz_convert("UTC").ceil("D")
    last = pd.Timestamp(end).tz_convert("UTC").floor("D")
    return [day.strftime("%Y-%m-%d") for day in pd.date_range(first, last, freq="D", inclusive="left")]


def create_experiment(
    base: Path, *, days: int, symbol: str, experiment_id: str | None = None,
    frozen_config: Path | None = None, now: datetime | None = None,
) -> Path:
    if days < 1:
        raise ValueError("days must be positive")
    started = (now or datetime.now(UTC)).astimezone(UTC)
    experiment_id = experiment_id or started.strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    if not experiment_id.replace("-", "").replace("_", "").isalnum():
        raise ValueError("experiment_id must be an alphanumeric path component")
    root = Path(base) / "long" / experiment_id
    root.mkdir(parents=True, exist_ok=False)
    (root / "sessions").mkdir()
    first_full_day = (started.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1))
    frozen = frozen_config or PROJECT_ROOT / "configs" / "confirmatory_ridge30.yaml"
    manifest = root / "experiment.json"
    atomic_write_json({
        "schema_version": 2, "experiment_id": experiment_id, "symbol": symbol,
        "exchange": "bybit", "started_utc": started.isoformat(),
        "first_complete_day_utc": first_full_day.isoformat(),
        "planned_end_utc": (first_full_day + timedelta(days=days)).isoformat(),
        "target_complete_days": days, "frozen_config_sha256": sha256_file(frozen),
        "git_commit": git_commit(PROJECT_ROOT), "storage": "immutable session parts; 15-minute/100000-row rotation",
    }, manifest)
    return manifest


def resolve_experiment(experiment_id: str | None = None, *, base: Path | None = None) -> Path:
    root = Path(base) if base is not None else LONG_ROOT
    if experiment_id:
        result = root / experiment_id / "experiment.json"
    else:
        options = sorted(root.glob("*/experiment.json")) if root.exists() else []
        if not options:
            raise FileNotFoundError(f"no long experiments under {root}")
        result = options[-1]
    if not result.is_file():
        raise FileNotFoundError(result)
    return result


def _registered_ids(manifest: Path) -> list[str]:
    return [path.stem for path in sorted((manifest.parent / "sessions").glob("*.json"))]


def _last_raw_timestamp(path: Path) -> pd.Timestamp | None:
    parts = sorted(path.glob("part-*.parquet")) if path.exists() else []
    if not parts:
        return None
    values = pd.read_parquet(parts[-1], columns=["recv_utc_ns"])["recv_utc_ns"]
    return pd.Timestamp(int(values.max()), unit="ns", tz="UTC") if len(values) else None


def recording_sessions(manifest: Path, raw_root: Path) -> pd.DataFrame:
    """Recoverable session inventory, including crashed sessions with no end manifest."""
    rows: list[dict[str, Any]] = []
    for session_id in _registered_ids(manifest):
        directory = Path(raw_root) / "sessions" / session_id
        start_file, end_file = directory / "session_start.json", directory / "session_end.json"
        start = json.loads(start_file.read_text(encoding="utf-8")) if start_file.exists() else {}
        end = json.loads(end_file.read_text(encoding="utf-8")) if end_file.exists() else {}
        first_ns = start.get("start_utc_ns")
        first = pd.Timestamp(int(first_ns), unit="ns", tz="UTC") if first_ns is not None else pd.NaT
        last_ns = end.get("end_utc_ns")
        if last_ns is not None:
            last = pd.Timestamp(int(last_ns), unit="ns", tz="UTC")
        else:
            candidates = [_last_raw_timestamp(Path(raw_root) / kind / session_id) for kind in ("orderbook", "trades", "aux")]
            last = max((value for value in candidates if value is not None), default=pd.NaT)
        files, compressed, raw_bytes, raw_rows = 0, 0, 0, 0
        for kind in ("orderbook", "trades", "aux"):
            for part in (Path(raw_root) / kind / session_id).glob("part-*.parquet"):
                files += 1
                compressed += part.stat().st_size
                metadata = pq.read_metadata(part)
                raw_rows += metadata.num_rows
                raw_bytes += sum(metadata.row_group(i).total_byte_size for i in range(metadata.num_row_groups))
        counters = end.get("counters", {})
        rows.append({
            "experiment_id": manifest.parent.name, "session_id": session_id,
            "start": first, "end": last, "status": end.get("status", "interrupted"),
            "process_stop_reason": end.get("error") or end.get("status", "process_exit_without_manifest"),
            "duration_s": max(0.0, (last - first).total_seconds()) if pd.notna(first) and pd.notna(last) else 0.0,
            "reconnect_count": int(counters.get("reconnect_count", 0)),
            "sequence_gap_count": int(counters.get("local_loss_count", 0) + counters.get("book_invalidation_count", 0)),
            "stale_count": int(counters.get("stale_connection_count", 0)),
            "message_count": int(counters.get("wire_message_count", 0)),
            "trade_count": int(counters.get("trade_count", 0)),
            "book_message_count": int(counters.get("orderbook_message_count", 0)),
            "files_written": files, "raw_bytes": raw_bytes, "compressed_bytes": compressed,
            "raw_rows": raw_rows, "software_version": start.get("host_clock", {}).get("python"),
            "git_commit": json.loads(manifest.read_text(encoding="utf-8")).get("git_commit"),
        })
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    table = table.sort_values("start", kind="stable").reset_index(drop=True)
    table["downtime_before_s"] = 0.0
    for index in range(1, len(table)):
        if pd.notna(table.loc[index - 1, "end"]) and pd.notna(table.loc[index, "start"]):
            table.loc[index, "downtime_before_s"] = max(0.0, (table.loc[index, "start"] - table.loc[index - 1, "end"]).total_seconds())
    return table


def observed_storage_estimate(config: dict[str, Any], days: int) -> dict[str, float]:
    """Base the estimate on the actual original 24-hour capture if present."""
    reference = PROJECT_ROOT / "runs" / "2026-09-12_070710_btcusdt" / "dataset_manifest.json"
    raw_per_day = 0.85 * 1024**3
    processed_per_day = 0.05 * 1024**3
    if reference.exists():
        source = json.loads(reference.read_text(encoding="utf-8"))
        capture = source["capture_session"]
        raw_per_day = sum(
            path.stat().st_size
            for kind in ("orderbook", "trades", "aux")
            for path in (project_path(config, "raw_dir") / kind / capture).glob("part-*.parquet")
        ) or raw_per_day
        processed_per_day = Path(source["research_table"]).stat().st_size if Path(source["research_table"]).exists() else processed_per_day
    total = days * (raw_per_day + processed_per_day)
    return {"raw_gb_per_day": raw_per_day / 1e9, "processed_gb_per_day": processed_per_day / 1e9,
            "expected_gb": total / 1e9, "expected_7day_gb": 7 * (raw_per_day + processed_per_day) / 1e9,
            "expected_14day_gb": 14 * (raw_per_day + processed_per_day) / 1e9}


def collect_long(
    config: dict[str, Any], frozen: dict[str, Any], *, days: int | None,
    resume: bool = False, experiment_id: str | None = None,
    progress: Callable[[dict[str, Any]], None] | None = None, progress_interval_s: float = 30.0,
    smoke_seconds: float | None = None,
) -> Path:
    raw_root = project_path(config, "raw_dir")
    if resume:
        manifest = resolve_experiment(experiment_id)
    else:
        manifest = create_experiment(PROJECT_ROOT / "data", days=days or 14,
                                     symbol=config["project"]["symbol"], experiment_id=experiment_id)
    experiment = json.loads(manifest.read_text(encoding="utf-8"))
    if days is not None and days != int(experiment["target_complete_days"]):
        raise ValueError("--days must match the immutable experiment target when resuming")
    if sha256_file(PROJECT_ROOT / "configs" / "confirmatory_ridge30.yaml") != experiment["frozen_config_sha256"]:
        raise ValueError("frozen confirmatory config changed since experiment creation")
    estimate = observed_storage_estimate(config, int(experiment["target_complete_days"]))
    reserve = int(float(frozen["stage2"]["disk_reserve_gb"]) * 1e9)
    free = shutil.disk_usage(raw_root).free if raw_root.exists() else shutil.disk_usage(raw_root.parent).free
    print(f"Experiment {experiment['experiment_id']} | observed raw {estimate['raw_gb_per_day']:.2f} GB/day, processed {estimate['processed_gb_per_day']:.2f} GB/day; 7d {estimate['expected_7day_gb']:.1f} GB, 14d {estimate['expected_14day_gb']:.1f} GB; free {free/1e9:.1f} GB", flush=True)
    if free < estimate["expected_gb"] + reserve:
        raise OSError("insufficient free disk space for observed-duration estimate plus reserve")
    long_config = replace(CollectorConfig.from_mapping(config), raw_root=raw_root,
                          part_seconds=float(frozen["stage2"]["long_part_seconds"]),
                          part_rows=int(frozen["stage2"]["long_part_rows"]))
    planned_end = datetime.fromisoformat(experiment["planned_end_utc"])
    smoke_end = datetime.now(UTC) + timedelta(seconds=smoke_seconds) if smoke_seconds else None
    while datetime.now(UTC) < planned_end:
        now = datetime.now(UTC)
        if smoke_end and now >= smoke_end:
            break
        next_midnight = now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
        until = min(next_midnight, planned_end, smoke_end) if smoke_end else min(next_midnight, planned_end)
        duration = (until - now).total_seconds()
        if duration < 0.1:
            time.sleep(0.2)
            continue
        session_id = now.strftime("%Y%m%dT%H%M%S.%fZ") + "-" + uuid.uuid4().hex[:12]
        atomic_write_json({"session_id": session_id, "experiment_id": experiment["experiment_id"],
                           "registered_utc": now.isoformat()}, manifest.parent / "sessions" / f"{session_id}.json")

        def heartbeat(item: dict[str, Any]) -> None:
            item = {**item, "experiment_id": experiment["experiment_id"], "reported_utc": datetime.now(UTC).isoformat()}
            temporary = manifest.parent / f".heartbeat.{os.getpid()}.tmp"
            temporary.write_text(json.dumps(item, ensure_ascii=False), encoding="utf-8")
            os.replace(temporary, manifest.parent / "heartbeat.json")
            if progress:
                progress(item)

        print(f"New immutable recording session {session_id}; scheduled stop {until.isoformat()}", flush=True)
        collect_blocking(long_config, duration_s=duration, session_id=session_id,
                         progress_callback=heartbeat, progress_interval_s=progress_interval_s,
                         disk_reserve_bytes=reserve)
    return manifest


def experiment_status(manifest: Path, raw_root: Path) -> dict[str, Any]:
    meta = json.loads(manifest.read_text(encoding="utf-8"))
    sessions = recording_sessions(manifest, raw_root)
    heartbeat_file = manifest.parent / "heartbeat.json"
    heartbeat = json.loads(heartbeat_file.read_text(encoding="utf-8")) if heartbeat_file.exists() else {}
    recent = pd.Timestamp(heartbeat["reported_utc"]) if heartbeat else None
    alive = bool(recent is not None and (pd.Timestamp.now(tz="UTC") - recent).total_seconds() < 2 * 30 + 15 and heartbeat.get("status") not in {"complete", "failed", "cancelled"})
    return {
        "experiment_id": meta["experiment_id"], "planned_end_utc": meta["planned_end_utc"],
        "status": "active" if alive else "stopped", "sessions": len(sessions),
        "collected_s": (float(sessions["duration_s"].sum()) if not sessions.empty else 0.0)
                       + (float(heartbeat.get("elapsed_s", 0.0)) if alive and not sessions.empty
                          and sessions.iloc[-1]["status"] == "interrupted" else 0.0),
        "downtime_s": float(sessions["downtime_before_s"].sum()) if not sessions.empty else 0.0,
        "compressed_bytes": int(sessions["compressed_bytes"].sum()) if not sessions.empty else 0,
        "buffered_rows": int(heartbeat.get("buffered_rows", 0)) if alive else 0,
        "live_elapsed_s": float(heartbeat.get("elapsed_s", 0)) if alive else 0.0,
        "invalid_periods_observed": int(sessions["sequence_gap_count"].sum() + sessions["stale_count"].sum()) if not sessions.empty else 0,
        "reconnects": int(sessions["reconnect_count"].sum()) + (int(heartbeat.get("reconnects", 0)) if alive else 0),
        "last_message_utc": heartbeat.get("last_message_utc"),
        "latest_valid_book_utc": heartbeat.get("last_valid_book_utc"),
        "message_rate_hz": heartbeat.get("message_rate_hz") if alive else None,
        "book_valid": heartbeat.get("book_valid") if alive else None,
        "free_disk_gb": shutil.disk_usage(raw_root).free / 1e9,
    }


__all__ = ["collect_long", "completed_utc_days", "create_experiment", "experiment_status", "observed_storage_estimate", "recording_sessions", "resolve_experiment"]
