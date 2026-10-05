from __future__ import annotations

import json

import pandas as pd

from src.build_dataset import _auxiliary_grid, replay_order_book
from src.collect import CollectorConfig, CollectorSession


def test_collector_raw_parts_replay_by_receive_time(tmp_path) -> None:
    session = CollectorSession(
        CollectorConfig(raw_root=tmp_path, part_rows=1, part_seconds=60),
        session_id="toy",
    )
    session.connection_id = "connection-1"
    base = pd.Timestamp("2026-01-01T00:00:00Z").value
    snapshot = {
        "topic": "orderbook.50.BTCUSDT", "type": "snapshot", "ts": 1_000,
        "cts": 999, "data": {"s": "BTCUSDT", "u": 1, "seq": 10,
        "b": [["100", "2"]], "a": [["101", "3"]]},
    }
    delta = {
        "topic": "orderbook.50.BTCUSDT", "type": "delta", "ts": 2_000,
        "cts": 1_999, "data": {"s": "BTCUSDT", "u": 3, "seq": 12,
        "b": [["100", "4"]], "a": []},
    }
    session.handle_wire(json.dumps(snapshot), recv_utc_ns=base + 200_000_000, recv_monotonic_ns=1)
    session.handle_wire(json.dumps(delta), recv_utc_ns=base + 1_200_000_000, recv_monotonic_ns=2)
    progress = session.progress_snapshot()
    assert progress["wire_messages"] == 2
    assert progress["book_messages"] == 2
    assert progress["published_parts"] >= 2
    assert progress["saved_bytes"] > 0
    assert progress["book_valid"] is True
    session.close("complete")

    states, ofi, counts = replay_order_book(
        sorted((tmp_path / "orderbook" / "toy").glob("part-*.parquet")), symbol="BTCUSDT"
    )
    assert states.iloc[0]["bid_size_01"] == 2
    assert states.iloc[1]["bid_size_01"] == 4
    assert counts["sequence_diagnostics"] == 1  # a forward jump is diagnostic, not invented loss
    assert len(ofi) == 1


def test_shared_trade_sequence_is_not_a_duplicate_and_ticker_delta_is_sparse(tmp_path) -> None:
    session = CollectorSession(
        CollectorConfig(raw_root=tmp_path, part_rows=100, part_seconds=60),
        session_id="wire",
    )
    session.connection_id = "connection-1"
    base = pd.Timestamp("2026-01-01T00:00:00Z").value
    trades = {
        "topic": "publicTrade.BTCUSDT", "type": "snapshot", "ts": 1_000,
        "data": [
            {"T": 999, "s": "BTCUSDT", "S": "Buy", "p": "100", "v": "1", "i": "a", "seq": 7},
            {"T": 999, "s": "BTCUSDT", "S": "Sell", "p": "101", "v": "2", "i": "b", "seq": 7},
        ],
    }
    ticker = {
        "topic": "tickers.BTCUSDT", "type": "snapshot", "ts": 1_000,
        "data": {"symbol": "BTCUSDT", "bid1Price": "100", "ask1Price": "101", "openInterest": "10"},
    }
    sparse = {
        "topic": "tickers.BTCUSDT", "type": "delta", "ts": 2_000,
        "data": {"symbol": "BTCUSDT", "lastPrice": "100.5"},
    }
    session.handle_wire(json.dumps(trades), recv_utc_ns=base + 100_000_000, recv_monotonic_ns=1)
    session.handle_wire(json.dumps(trades), recv_utc_ns=base + 200_000_000, recv_monotonic_ns=2)
    session.handle_wire(json.dumps(ticker), recv_utc_ns=base + 300_000_000, recv_monotonic_ns=3)
    session.handle_wire(json.dumps(sparse), recv_utc_ns=base + 1_300_000_000, recv_monotonic_ns=4)
    session.close("complete")

    captured = pd.concat(pd.read_parquet(path) for path in sorted((tmp_path / "trades" / "wire").glob("part-*.parquet")))
    assert captured.iloc[:2]["duplicate"].eq(False).all()
    assert captured.iloc[2:]["duplicate"].eq(True).all()
    assert captured.iloc[:2]["cross_seq"].eq(7).all()

    grid = pd.DataFrame({"timestamp": pd.date_range("2026-01-01T00:00:01Z", periods=2, freq="1s", tz="UTC")})
    aux = _auxiliary_grid(sorted((tmp_path / "aux" / "wire").glob("part-*.parquet")), grid, interval_s=1)
    assert aux["open_interest"].tolist() == [10.0, 10.0]
    assert aux["ticker_bid1"].tolist() == [100.0, 100.0]


def test_session_id_is_immutable(tmp_path) -> None:
    first = CollectorSession(CollectorConfig(raw_root=tmp_path), session_id="same")
    first.close("complete")
    try:
        CollectorSession(CollectorConfig(raw_root=tmp_path), session_id="same")
    except FileExistsError:
        pass
    else:
        raise AssertionError("an existing raw session must never be reused")
