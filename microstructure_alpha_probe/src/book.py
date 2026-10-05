"""Conservative Bybit V5 order-book reconstruction.

Bybit documents update and cross-sequence identifiers, but it doesn't document
that either counter is contiguous.  Consequently, forward jumps are retained as
diagnostics and don't invalidate the book by default.  Proven local/transport
loss, conflicting identities, decreasing identifiers, malformed updates, and
an invalid reconstructed book do invalidate it until the next snapshot.
"""

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any, Mapping, Sequence


class BookMessageError(ValueError):
    """A wire message cannot be applied safely."""


@dataclass(frozen=True, slots=True)
class BookUpdateResult:
    """Outcome recorded alongside one raw order-book message."""

    state_action: str
    book_valid_after: bool
    duplicate: bool = False
    sequence_anomaly: str | None = None
    error: str | None = None
    ofi: float | None = None


def _as_int(value: Any, field: str) -> int:
    if isinstance(value, bool):
        raise BookMessageError(f"{field} must be an integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise BookMessageError(f"{field} must be an integer") from exc
    if parsed < 0:
        raise BookMessageError(f"{field} must be non-negative")
    return parsed


def _as_decimal(value: Any, field: str, *, allow_zero: bool = False) -> Decimal:
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise BookMessageError(f"{field} is not numeric") from exc
    if not parsed.is_finite() or parsed < 0 or (not allow_zero and parsed == 0):
        relation = "non-negative" if allow_zero else "positive"
        raise BookMessageError(f"{field} must be finite and {relation}")
    return parsed


def _canonical_fingerprint(message: Mapping[str, Any]) -> str:
    relevant = {
        "type": message.get("type"),
        "ts": message.get("ts"),
        "cts": message.get("cts"),
        "data": message.get("data"),
    }
    raw = json.dumps(
        relevant, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


class OrderBook:
    """Reconstruct one depth-limited order book from Bybit snapshot/delta data."""

    def __init__(
        self,
        *,
        symbol: str = "BTCUSDT",
        identity_cache_size: int = 20_000,
        require_contiguous_ids: bool = False,
    ) -> None:
        if identity_cache_size < 1:
            raise ValueError("identity_cache_size must be positive")
        self.symbol = symbol.upper()
        self.identity_cache_size = identity_cache_size
        self.require_contiguous_ids = require_contiguous_ids
        self.bids: dict[Decimal, Decimal] = {}
        self.asks: dict[Decimal, Decimal] = {}
        self.valid = False
        self.invalid_reason: str | None = "waiting_for_snapshot"
        self.last_update_id: int | None = None
        self.last_cross_seq: int | None = None
        self.last_exchange_ts_ms: int | None = None
        self.last_matching_ts_ms: int | None = None
        self.resynchronizations = 0
        self._has_snapshot = False
        self._seen: OrderedDict[tuple[int, int], str] = OrderedDict()

    @property
    def best_bid(self) -> tuple[Decimal, Decimal] | None:
        if not self.bids:
            return None
        price = max(self.bids)
        return price, self.bids[price]

    @property
    def best_ask(self) -> tuple[Decimal, Decimal] | None:
        if not self.asks:
            return None
        price = min(self.asks)
        return price, self.asks[price]

    def top(
        self, depth: int = 20
    ) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
        """Return numeric bids descending and asks ascending."""
        if depth < 1:
            raise ValueError("depth must be positive")
        bids = sorted(self.bids.items(), reverse=True)[:depth]
        asks = sorted(self.asks.items())[:depth]
        return (
            [(float(price), float(size)) for price, size in bids],
            [(float(price), float(size)) for price, size in asks],
        )

    def invalidate(self, reason: str) -> BookUpdateResult:
        """Mark state unusable; only a subsequent snapshot can restore it."""
        self.valid = False
        self.invalid_reason = reason
        return BookUpdateResult(
            state_action="invalidated", book_valid_after=False, error=reason
        )

    def mark_transport_loss(self, reason: str = "transport_loss") -> BookUpdateResult:
        return self.invalidate(reason)

    def process(
        self,
        message: Mapping[str, Any],
        *,
        payload_hash: str | None = None,
    ) -> BookUpdateResult:
        """Apply one decoded raw order-book message without guessing over loss."""
        try:
            return self._process(message, payload_hash=payload_hash)
        except (BookMessageError, KeyError, TypeError) as exc:
            reason = f"malformed_book_message: {exc}"
            self.invalidate(reason)
            return BookUpdateResult(
                state_action="invalidated",
                book_valid_after=False,
                error=reason,
            )

    def _process(
        self,
        message: Mapping[str, Any],
        *,
        payload_hash: str | None,
    ) -> BookUpdateResult:
        if not isinstance(message, Mapping):
            raise BookMessageError("message must be an object")
        message_type = message.get("type")
        if message_type not in {"snapshot", "delta"}:
            raise BookMessageError("type must be snapshot or delta")
        data = message.get("data")
        if not isinstance(data, Mapping):
            raise BookMessageError("data must be an object")
        if str(data.get("s", "")).upper() != self.symbol:
            raise BookMessageError(f"unexpected symbol {data.get('s')!r}")
        update_id = _as_int(data.get("u"), "data.u")
        cross_seq = _as_int(data.get("seq"), "data.seq")
        exchange_ts = _as_int(message.get("ts"), "ts")
        matching_ts = _as_int(message.get("cts"), "cts")
        fingerprint = payload_hash or _canonical_fingerprint(message)
        identity = (update_id, cross_seq)

        # Every snapshot is authoritative.  An exact replay can be ignored, while
        # a different snapshot starts a new synchronization epoch as documented.
        if message_type == "snapshot":
            if self.valid and self._seen.get(identity) == fingerprint:
                return BookUpdateResult(
                    state_action="duplicate_snapshot",
                    book_valid_after=True,
                    duplicate=True,
                )
            return self._apply_snapshot(
                data,
                identity=identity,
                fingerprint=fingerprint,
                exchange_ts=exchange_ts,
                matching_ts=matching_ts,
            )

        previous_fingerprint = self._seen.get(identity)
        if previous_fingerprint is not None:
            if previous_fingerprint == fingerprint:
                return BookUpdateResult(
                    state_action="duplicate_delta",
                    book_valid_after=self.valid,
                    duplicate=True,
                )
            reason = f"conflicting_identity:u={update_id},seq={cross_seq}"
            self.invalidate(reason)
            return BookUpdateResult(
                state_action="invalidated",
                book_valid_after=False,
                sequence_anomaly="conflicting_identity",
                error=reason,
            )

        if not self.valid:
            return BookUpdateResult(
                state_action="delta_ignored_unsynced",
                book_valid_after=False,
                error=self.invalid_reason or "waiting_for_snapshot",
            )

        diagnostics: list[str] = []
        if self.last_update_id is not None:
            if update_id < self.last_update_id:
                diagnostics.append("decreasing_update_id")
            elif update_id > self.last_update_id + 1:
                diagnostics.append("update_id_jump")
        if self.last_cross_seq is not None:
            if cross_seq < self.last_cross_seq:
                diagnostics.append("decreasing_cross_seq")
            elif cross_seq > self.last_cross_seq + 1:
                diagnostics.append("cross_seq_jump")

        decreasing = any(item.startswith("decreasing_") for item in diagnostics)
        jump = any(item.endswith("_jump") for item in diagnostics)
        if decreasing or (jump and self.require_contiguous_ids):
            reason = ",".join(diagnostics)
            self.invalidate(reason)
            return BookUpdateResult(
                state_action="invalidated",
                book_valid_after=False,
                sequence_anomaly=reason,
                error=reason,
            )

        old_bid, old_ask = self.best_bid, self.best_ask
        bids = dict(self.bids)
        asks = dict(self.asks)
        bid_updates = self._parse_side(data.get("b"), "bids", snapshot=False)
        ask_updates = self._parse_side(data.get("a"), "asks", snapshot=False)
        self._apply_side(bids, bid_updates, "bids")
        self._apply_side(asks, ask_updates, "asks")
        self._validate_book(bids, asks)

        self.bids, self.asks = bids, asks
        self.last_update_id = update_id
        self.last_cross_seq = cross_seq
        self.last_exchange_ts_ms = exchange_ts
        self.last_matching_ts_ms = matching_ts
        self._remember(identity, fingerprint)
        return BookUpdateResult(
            state_action="delta_applied",
            book_valid_after=True,
            sequence_anomaly=",".join(diagnostics) or None,
            ofi=self._ofi(old_bid, old_ask, self.best_bid, self.best_ask),
        )

    def _apply_snapshot(
        self,
        data: Mapping[str, Any],
        *,
        identity: tuple[int, int],
        fingerprint: str,
        exchange_ts: int,
        matching_ts: int,
    ) -> BookUpdateResult:
        bid_updates = self._parse_side(data.get("b"), "bids", snapshot=True)
        ask_updates = self._parse_side(data.get("a"), "asks", snapshot=True)
        bid_prices = [price for price, _ in bid_updates]
        ask_prices = [price for price, _ in ask_updates]
        if bid_prices != sorted(bid_prices, reverse=True):
            raise BookMessageError("snapshot bids are not descending")
        if ask_prices != sorted(ask_prices):
            raise BookMessageError("snapshot asks are not ascending")
        bids, asks = dict(bid_updates), dict(ask_updates)
        self._validate_book(bids, asks)

        action = "resnapshot" if self._has_snapshot else "snapshot"
        if self._has_snapshot:
            self.resynchronizations += 1
        self.bids, self.asks = bids, asks
        self.valid = True
        self.invalid_reason = None
        self.last_update_id, self.last_cross_seq = identity
        self.last_exchange_ts_ms = exchange_ts
        self.last_matching_ts_ms = matching_ts
        self._has_snapshot = True
        self._seen.clear()
        self._remember(identity, fingerprint)
        return BookUpdateResult(state_action=action, book_valid_after=True)

    @staticmethod
    def _parse_side(
        entries: Any, side: str, *, snapshot: bool
    ) -> list[tuple[Decimal, Decimal]]:
        if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)):
            raise BookMessageError(f"data.{side[0]} must be an array")
        parsed: list[tuple[Decimal, Decimal]] = []
        seen: set[Decimal] = set()
        for index, entry in enumerate(entries):
            if (
                not isinstance(entry, Sequence)
                or isinstance(entry, (str, bytes))
                or len(entry) != 2
            ):
                raise BookMessageError(f"{side}[{index}] must be [price,size]")
            price = _as_decimal(entry[0], f"{side}[{index}].price")
            size = _as_decimal(
                entry[1], f"{side}[{index}].size", allow_zero=not snapshot
            )
            if price in seen:
                raise BookMessageError(f"duplicate {side} price {price}")
            seen.add(price)
            parsed.append((price, size))
        return parsed

    @staticmethod
    def _apply_side(
        side: dict[Decimal, Decimal],
        updates: list[tuple[Decimal, Decimal]],
        side_name: str,
    ) -> None:
        for price, size in updates:
            if size == 0:
                if price not in side:
                    raise BookMessageError(
                        f"cannot delete absent {side_name} price {price}"
                    )
                del side[price]
            else:
                side[price] = size

    @staticmethod
    def _validate_book(
        bids: Mapping[Decimal, Decimal], asks: Mapping[Decimal, Decimal]
    ) -> None:
        if not bids or not asks:
            raise BookMessageError("book must have both bid and ask levels")
        if max(bids) >= min(asks):
            raise BookMessageError("crossed_or_locked_book")

    def _remember(self, identity: tuple[int, int], fingerprint: str) -> None:
        self._seen[identity] = fingerprint
        self._seen.move_to_end(identity)
        while len(self._seen) > self.identity_cache_size:
            self._seen.popitem(last=False)

    @staticmethod
    def _ofi(
        old_bid: tuple[Decimal, Decimal] | None,
        old_ask: tuple[Decimal, Decimal] | None,
        new_bid: tuple[Decimal, Decimal] | None,
        new_ask: tuple[Decimal, Decimal] | None,
    ) -> float | None:
        """Cont-style level-one order-flow imbalance for one book event."""
        if None in {old_bid, old_ask, new_bid, new_ask}:
            return None
        assert old_bid is not None and old_ask is not None
        assert new_bid is not None and new_ask is not None
        old_bp, old_bq = old_bid
        old_ap, old_aq = old_ask
        new_bp, new_bq = new_bid
        new_ap, new_aq = new_ask
        value = Decimal(0)
        if new_bp >= old_bp:
            value += new_bq
        if new_bp <= old_bp:
            value -= old_bq
        if new_ap <= old_ap:
            value -= new_aq
        if new_ap >= old_ap:
            value += old_aq
        return float(value)


__all__ = ["BookMessageError", "BookUpdateResult", "OrderBook"]
