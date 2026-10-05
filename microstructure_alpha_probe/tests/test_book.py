from __future__ import annotations

from src.book import OrderBook


def book_message(
    message_type: str,
    *,
    update_id: int,
    seq: int,
    bids: list[list[str]],
    asks: list[list[str]],
    ts: int = 1_000,
) -> dict:
    return {
        "topic": "orderbook.50.BTCUSDT",
        "type": message_type,
        "ts": ts,
        "cts": ts - 1,
        "data": {
            "s": "BTCUSDT",
            "b": bids,
            "a": asks,
            "u": update_id,
            "seq": seq,
        },
    }


def initial_snapshot(update_id: int = 10, seq: int = 100) -> dict:
    return book_message(
        "snapshot",
        update_id=update_id,
        seq=seq,
        bids=[["100", "2"], ["99", "3"]],
        asks=[["101", "4"], ["102", "5"]],
    )


def test_snapshot_reconstructs_sorted_book() -> None:
    book = OrderBook()

    result = book.process(initial_snapshot())

    assert result.state_action == "snapshot"
    assert result.book_valid_after
    assert book.valid
    assert book.top(2) == (
        [(100.0, 2.0), (99.0, 3.0)],
        [(101.0, 4.0), (102.0, 5.0)],
    )
    assert book.last_update_id == 10
    assert book.last_cross_seq == 100


def test_delta_delete_insert_update_and_ofi() -> None:
    book = OrderBook()
    book.process(initial_snapshot())
    delta = book_message(
        "delta",
        update_id=11,
        seq=101,
        bids=[["100", "5"], ["98", "1"], ["99", "0"]],
        asks=[["101", "0"], ["100.5", "2"]],
        ts=1_001,
    )

    result = book.process(delta)

    assert result.state_action == "delta_applied"
    assert result.book_valid_after
    assert result.sequence_anomaly is None
    assert result.ofi == 1.0
    assert book.top(3) == (
        [(100.0, 5.0), (98.0, 1.0)],
        [(100.5, 2.0), (102.0, 5.0)],
    )


def test_forward_identifier_jumps_are_diagnostic_not_loss_by_default() -> None:
    book = OrderBook()
    book.process(initial_snapshot(update_id=100, seq=1_000))

    result = book.process(
        book_message(
            "delta",
            update_id=107,
            seq=1_025,
            bids=[["100", "2.5"]],
            asks=[],
        )
    )

    assert result.book_valid_after
    assert result.state_action == "delta_applied"
    assert result.sequence_anomaly == "update_id_jump,cross_seq_jump"
    assert book.valid


def test_strict_contiguous_mode_is_opt_in() -> None:
    book = OrderBook(require_contiguous_ids=True)
    book.process(initial_snapshot(update_id=100, seq=1_000))

    result = book.process(
        book_message(
            "delta",
            update_id=102,
            seq=1_002,
            bids=[["100", "2.5"]],
            asks=[],
        )
    )

    assert not result.book_valid_after
    assert result.state_action == "invalidated"


def test_exact_duplicate_is_ignored_but_conflicting_identity_invalidates() -> None:
    book = OrderBook()
    book.process(initial_snapshot())
    delta = book_message(
        "delta",
        update_id=11,
        seq=101,
        bids=[["100", "3"]],
        asks=[],
    )
    first = book.process(delta)
    duplicate = book.process(delta)
    conflict = book.process(
        book_message(
            "delta",
            update_id=11,
            seq=101,
            bids=[["100", "9"]],
            asks=[],
        )
    )

    assert first.book_valid_after
    assert duplicate.duplicate
    assert duplicate.state_action == "duplicate_delta"
    assert not conflict.book_valid_after
    assert conflict.sequence_anomaly == "conflicting_identity"
    assert book.top(1)[0] == [(100.0, 3.0)]


def test_decreasing_sequence_invalidates_until_a_fresh_snapshot() -> None:
    book = OrderBook()
    book.process(initial_snapshot())
    decreasing = book.process(
        book_message(
            "delta",
            update_id=11,
            seq=99,
            bids=[["100", "3"]],
            asks=[],
        )
    )
    ignored = book.process(
        book_message(
            "delta",
            update_id=12,
            seq=102,
            bids=[["100", "4"]],
            asks=[],
        )
    )
    recovered = book.process(initial_snapshot(update_id=1, seq=200))

    assert decreasing.state_action == "invalidated"
    assert decreasing.sequence_anomaly == "decreasing_cross_seq"
    assert ignored.state_action == "delta_ignored_unsynced"
    assert not ignored.book_valid_after
    assert recovered.state_action == "resnapshot"
    assert recovered.book_valid_after
    assert book.resynchronizations == 1


def test_explicit_transport_loss_invalidates_until_snapshot() -> None:
    book = OrderBook()
    book.process(initial_snapshot())

    invalidated = book.mark_transport_loss("socket_closed")
    ignored = book.process(
        book_message(
            "delta",
            update_id=11,
            seq=101,
            bids=[["100", "7"]],
            asks=[],
        )
    )

    assert invalidated.state_action == "invalidated"
    assert not book.valid
    assert ignored.state_action == "delta_ignored_unsynced"
    assert book.top(1)[0] == [(100.0, 2.0)]
    assert book.process(initial_snapshot(update_id=20, seq=200)).book_valid_after


def test_crossed_delta_is_atomic_and_invalidates() -> None:
    book = OrderBook()
    book.process(initial_snapshot())

    result = book.process(
        book_message(
            "delta",
            update_id=11,
            seq=101,
            bids=[],
            asks=[["101", "0"], ["99.5", "4"]],
        )
    )

    assert not result.book_valid_after
    assert "crossed_or_locked_book" in (result.error or "")
    # Invalid delta wasn't partially committed; the old state is retained only
    # for diagnosis and remains unusable until a snapshot.
    assert book.top(1)[1] == [(101.0, 4.0)]
    assert not book.valid


def test_bad_snapshot_never_becomes_valid() -> None:
    book = OrderBook()
    result = book.process(
        book_message(
            "snapshot",
            update_id=1,
            seq=1,
            bids=[["100", "1"], ["101", "1"]],
            asks=[["100.5", "1"]],
        )
    )

    assert not result.book_valid_after
    assert result.state_action == "invalidated"
    assert not book.valid
