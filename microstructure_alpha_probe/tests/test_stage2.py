from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.long_analysis import buffered_eligibility, nonoverlap_ic_by_phase, quote_at_or_after, stale_book_at
from src.long_collection import completed_utc_days, create_experiment, recording_sessions
from src.long_analysis import load_frozen_baseline
from src.targets import add_targets
from src.collect import AtomicParquetPartWriter


def test_resume_creates_distinct_sessions_and_preserves_recorded_downtime(tmp_path) -> None:
    manifest = create_experiment(tmp_path, days=7, symbol="BTCUSDT")
    session_root = tmp_path / "raw" / "sessions"
    for name, start, end in (("one", 0, 36), ("two", 37, 38)):
        directory = session_root / name
        directory.mkdir(parents=True)
        (manifest.parent / "sessions" / f"{name}.json").write_text(json.dumps({"session_id": name}), encoding="utf-8")
        (directory / "session_start.json").write_text(json.dumps({"start_utc_ns": start * 3_600_000_000_000}), encoding="utf-8")
        (directory / "session_end.json").write_text(json.dumps({"end_utc_ns": end * 3_600_000_000_000, "status": "complete", "counters": {"reconnect_count": 2}}), encoding="utf-8")
    sessions = recording_sessions(manifest, tmp_path / "raw")
    assert sessions["session_id"].tolist() == ["one", "two"]
    assert sessions["downtime_before_s"].tolist() == [0, 3600]
    assert sessions["reconnect_count"].tolist() == [2, 2]


def test_buffer_excludes_features_and_whole_target_path() -> None:
    times = pd.date_range("2026-01-01", periods=16, freq="1s", tz="UTC")
    source = pd.Series(False, index=times)
    source.iloc[7:9] = True
    eligibility, intervals = buffered_eligibility(times, source, before_s=2, after_s=2, horizons_s=[2, 5])
    assert intervals.iloc[0]["start"] == times[7]
    assert intervals.iloc[0]["end"] == times[8]
    assert eligibility.loc[5, "buffered_invalid"]
    assert eligibility.loc[10, "buffered_invalid"]
    assert not eligibility.loc[11, "buffered_invalid"]
    assert not eligibility.loc[3, "eligible_2s"]  # target reaches buffered interval
    assert eligibility.loc[11, "eligible_2s"]
    assert not eligibility.loc[11, "eligible_5s"]  # future endpoint outside the grid


def test_every_nonoverlapping_phase_is_reported_on_absolute_utc_grid() -> None:
    frame = pd.DataFrame({"timestamp": pd.date_range("2026-01-01", periods=12, freq="1s", tz="UTC"),
                          "prediction": np.arange(12, dtype=float), "actual": np.arange(12, dtype=float)})
    phases = nonoverlap_ic_by_phase(frame, horizon_s=3)
    assert phases["phase"].tolist() == [0, 1, 2]
    assert phases["n"].tolist() == [4, 4, 4]
    assert np.allclose(phases["spearman"], 1.0)


def test_raw_event_latency_uses_stale_information_and_delayed_execution() -> None:
    base = pd.Timestamp("2026-01-01T00:00:00Z")
    ticks = pd.DataFrame({"timestamp": [base + pd.Timedelta(milliseconds=x) for x in (0, 200, 500, 800)],
                          "book_valid": [True] * 4, "best_bid": [99, 100, 101, 102],
                          "best_ask": [101, 102, 103, 104], "obi_1": [-0.5, 0.0, 0.5, 0.9]})
    decision = base + pd.Timedelta(milliseconds=500)
    assert stale_book_at(ticks, decision, latency_ms=250)["obi_1"] == 0.0
    assert quote_at_or_after(ticks, decision, latency_ms=250)["best_ask"] == 104
    assert quote_at_or_after(ticks, decision, latency_ms=1000, max_wait_ms=100) is None


def test_delayed_execution_skips_invalid_raw_quote_within_wait_budget() -> None:
    base = pd.Timestamp("2026-01-01T00:00:00Z")
    ticks = pd.DataFrame({"timestamp": [base + pd.Timedelta(milliseconds=x) for x in (300, 350, 650)],
                          "book_valid": [False, True, True], "best_ask": [100, 101, 102]})
    assert quote_at_or_after(ticks, base, latency_ms=250, max_wait_ms=150)["best_ask"] == 101
    assert quote_at_or_after(ticks, base, latency_ms=250, max_wait_ms=50) is None


def test_long_horizon_target_requires_exact_future_and_complete_path() -> None:
    times = pd.date_range("2026-01-01", periods=905, freq="1s", tz="UTC")
    frame = pd.DataFrame({"timestamp": times, "mid": 100 + np.arange(905) / 1000,
                          "best_bid": 99 + np.arange(905) / 1000,
                          "best_ask": 101 + np.arange(905) / 1000,
                          "book_valid": True, "gap_or_resync_interval": False})
    out = add_targets(frame, horizons_s=[900])
    assert out.loc[0, "target_path_valid_900s"]
    assert not out.loc[5, "target_path_valid_900s"]
    frame.loc[450, "gap_or_resync_interval"] = True
    out = add_targets(frame, horizons_s=[900])
    assert not out.loc[0, "target_path_valid_900s"]


def test_completed_utc_days_exclude_partial_boundaries() -> None:
    days = completed_utc_days(pd.Timestamp("2026-01-01T12:00Z"), pd.Timestamp("2026-01-04T02:00Z"))
    assert days == ["2026-01-02", "2026-01-03"]


def test_frozen_config_detects_changed_source_artifact(tmp_path) -> None:
    config = tmp_path / "frozen.yaml"
    config.write_text("source_run: old\nsource_decision_lock_sha256: deadbeef\nsource_dataset_sha256: deadbeef\n", encoding="utf-8")
    with pytest.raises(ValueError, match="frozen.*hash|hash.*frozen"):
        load_frozen_baseline(config, tmp_path)


def test_chunk_rotation_publishes_immutable_parts(tmp_path) -> None:
    writer = AtomicParquetPartWriter(tmp_path / "aux", "aux", max_rows=2, max_seconds=900)
    for number in (1, 2, 3):
        writer.append({"local_message_no": number, "recv_utc_ns": number})
    assert len(writer.parts) == 1
    first = writer.directory / writer.parts[0]
    saved = first.read_bytes()
    writer.close()
    assert len(writer.parts) == 2
    assert first.read_bytes() == saved
    assert [len(pd.read_parquet(writer.directory / part)) for part in writer.parts] == [2, 1]
