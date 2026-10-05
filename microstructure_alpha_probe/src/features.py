"""Causal microstructure features on a complete UTC research grid.

Inputs are deliberately plain pandas frames. ``timestamp`` always means local
availability time: exchange timestamps may be retained as diagnostics, but they
never decide whether an observation is visible at a grid tick.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np
import pandas as pd


DEFAULT_DEPTHS = (1, 5, 10, 20)
DEFAULT_TRADE_WINDOWS_S = (1, 5, 10, 30, 60)
DEFAULT_OFI_WINDOWS_S = (1, 5, 10, 30)
DEFAULT_RETURN_LAGS_S = (1, 5, 10, 30, 60)
DEFAULT_VOLATILITY_WINDOWS_S = (10, 30, 60)


def _utc(values: object, name: str = "timestamp") -> pd.Series:
    try:
        converted = pd.to_datetime(values, utc=True, errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain parseable UTC timestamps") from exc
    return pd.Series(converted, index=getattr(values, "index", None), name=name)


def _utc_timestamp(value: object, name: str) -> pd.Timestamp:
    try:
        result = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a timestamp") from exc
    if result.tzinfo is None:
        result = result.tz_localize("UTC")
    else:
        result = result.tz_convert("UTC")
    return result


def _duration(value: object, name: str) -> pd.Timedelta:
    result = pd.Timedelta(seconds=float(value)) if isinstance(value, (int, float, np.number)) else pd.Timedelta(value)
    if result <= pd.Timedelta(0):
        raise ValueError(f"{name} must be positive")
    return result


def _seconds_label(value: object) -> str:
    seconds = _duration(value, "window").total_seconds()
    return f"{int(seconds)}s" if float(seconds).is_integer() else f"{seconds:g}s"


def _level_name(side: str, field: str, level: int) -> str:
    return f"{side}_{field}_{level:02d}"


def _expand_level_lists(frame: pd.DataFrame, max_depth: int) -> pd.DataFrame:
    """Accept either frozen-schema level columns or ``bids``/``asks`` lists."""
    result = frame.copy()
    for side, source in (("bid", "bids"), ("ask", "asks")):
        for level in range(1, max_depth + 1):
            price_name = _level_name(side, "price", level)
            size_name = _level_name(side, "size", level)
            # Tolerate non-zero-padded columns at the ingestion boundary, but
            # always emit the frozen zero-padded processed schema.
            legacy_price = f"{side}_price_{level}"
            legacy_size = f"{side}_size_{level}"
            if price_name not in result and legacy_price in result:
                result[price_name] = result[legacy_price]
            if size_name not in result and legacy_size in result:
                result[size_name] = result[legacy_size]
            if price_name not in result:
                if source in result:
                    result[price_name] = result[source].map(
                        lambda levels, i=level - 1: levels[i][0]
                        if isinstance(levels, (list, tuple)) and len(levels) > i
                        else np.nan
                    )
                else:
                    result[price_name] = np.nan
            if size_name not in result:
                if source in result:
                    result[size_name] = result[source].map(
                        lambda levels, i=level - 1: levels[i][1]
                        if isinstance(levels, (list, tuple)) and len(levels) > i
                        else np.nan
                    )
                else:
                    result[size_name] = np.nan
    return result


def _exact_lookup(
    values: pd.Series, timestamps: pd.Series, delta: pd.Timedelta
) -> np.ndarray:
    indexed = pd.Series(values.to_numpy(), index=pd.DatetimeIndex(timestamps))
    return indexed.reindex(pd.DatetimeIndex(timestamps) - delta).to_numpy()


def ofi_event(
    previous_bid_price: float,
    previous_bid_size: float,
    previous_ask_price: float,
    previous_ask_size: float,
    bid_price: float,
    bid_size: float,
    ask_price: float,
    ask_size: float,
) -> float:
    """Return standard top-of-book OFI for one atomic state transition."""
    values = np.asarray(
        [
            previous_bid_price,
            previous_bid_size,
            previous_ask_price,
            previous_ask_size,
            bid_price,
            bid_size,
            ask_price,
            ask_size,
        ],
        dtype=float,
    )
    if not np.isfinite(values).all() or (values[[0, 2, 4, 6]] <= 0).any() or (values[[1, 3, 5, 7]] < 0).any():
        return float("nan")

    if bid_price > previous_bid_price:
        bid_component = bid_size
    elif bid_price == previous_bid_price:
        bid_component = bid_size - previous_bid_size
    else:
        bid_component = -previous_bid_size

    if ask_price < previous_ask_price:
        ask_component = ask_size
    elif ask_price == previous_ask_price:
        ask_component = ask_size - previous_ask_size
    else:
        ask_component = -previous_ask_size
    return float(bid_component - ask_component)


def compute_event_ofi(
    book_events: pd.DataFrame,
    *,
    timestamp_col: str = "timestamp",
    valid_col: str = "book_valid",
    event_type_col: str = "type",
    reset_col: str | None = None,
    group_cols: Iterable[str] = ("session_id", "connection_id", "symbol"),
) -> pd.Series:
    """Calculate OFI on accepted event states without crossing a reset/session.

    ``book_events`` must contain the *post-update* L1 state in the frozen schema.
    Snapshots themselves get ``NaN``; the first valid delta after a snapshot may
    use that snapshot as its previous state.
    """
    required = {
        timestamp_col,
        "bid_price_01",
        "bid_size_01",
        "ask_price_01",
        "ask_size_01",
    }
    missing = required.difference(book_events.columns)
    if missing:
        raise ValueError(f"book_events is missing columns: {sorted(missing)}")
    if book_events.empty:
        return pd.Series(dtype=float, index=book_events.index, name="ofi_event")

    work = book_events.copy()
    work[timestamp_col] = _utc(work[timestamp_col], timestamp_col)
    work["_original_position"] = np.arange(len(work))
    groups = [column for column in group_cols if column in work.columns]
    work = work.sort_values([*groups, timestamp_col, "_original_position"], kind="stable")

    same_group = pd.Series(True, index=work.index)
    for column in groups:
        same_group &= work[column].eq(work[column].shift())

    names = ("bid_price_01", "bid_size_01", "ask_price_01", "ask_size_01")
    numeric = {name: pd.to_numeric(work[name], errors="coerce") for name in names}
    previous = {name: values.shift() for name, values in numeric.items()}
    current_matrix = np.column_stack([numeric[name].to_numpy(float) for name in names])
    previous_matrix = np.column_stack([previous[name].to_numpy(float) for name in names])

    pb, pbs, pa, pas = previous_matrix.T
    b, bs, a, ass = current_matrix.T
    bid_component = np.select(
        [b > pb, b == pb, b < pb], [bs, bs - pbs, -pbs], default=np.nan
    )
    ask_component = np.select(
        [a < pa, a == pa, a > pa], [ass, ass - pas, -pas], default=np.nan
    )
    values = bid_component - ask_component

    valid = same_group.to_numpy(bool)
    if valid_col in work:
        current_valid = work[valid_col].fillna(False).astype(bool)
        valid &= current_valid.to_numpy() & current_valid.shift(fill_value=False).to_numpy()
    finite = np.isfinite(current_matrix).all(axis=1) & np.isfinite(previous_matrix).all(axis=1)
    valid &= finite
    valid &= (current_matrix[:, [0, 2]] > 0).all(axis=1)
    valid &= (previous_matrix[:, [0, 2]] > 0).all(axis=1)
    valid &= (current_matrix[:, [1, 3]] >= 0).all(axis=1)
    valid &= (previous_matrix[:, [1, 3]] >= 0).all(axis=1)

    if reset_col and reset_col in work:
        valid &= ~work[reset_col].fillna(False).astype(bool).to_numpy()
    elif event_type_col in work:
        valid &= ~work[event_type_col].astype("string").str.lower().eq("snapshot").fillna(False).to_numpy()
    if "gap_or_resync_interval" in work:
        valid &= ~work["gap_or_resync_interval"].fillna(True).astype(bool).to_numpy()

    values[~valid] = np.nan
    restored = np.full(len(work), np.nan, dtype=float)
    restored[work["_original_position"].to_numpy(int)] = values
    return pd.Series(restored, index=book_events.index, name="ofi_event")


def _trade_base(
    trade_seconds: pd.DataFrame | None,
    grid_index: pd.DatetimeIndex,
    *,
    timestamp_col: str,
    frequency: pd.Timedelta,
) -> pd.DataFrame:
    columns = ("buy_volume", "sell_volume", "buy_count", "sell_count")
    if trade_seconds is None or trade_seconds.empty:
        return pd.DataFrame({**{name: np.nan for name in columns}, "coverage": 0.0}, index=grid_index)

    source = trade_seconds.copy()
    if timestamp_col not in source:
        raise ValueError(f"trade_seconds is missing {timestamp_col!r}")
    source[timestamp_col] = _utc(source[timestamp_col], timestamp_col)
    raw = {"taker_side", "quantity"}.issubset(source.columns)

    if raw:
        quantity = pd.to_numeric(source["quantity"], errors="coerce")
        side = source["taker_side"].astype("string").str.lower()
        if quantity.isna().any() or (quantity < 0).any() or (~side.isin(["buy", "sell"])).any():
            raise ValueError("raw trades require finite non-negative quantity and Buy/Sell side")
        source["buy_volume"] = quantity.where(side.eq("buy"), 0.0)
        source["sell_volume"] = quantity.where(side.eq("sell"), 0.0)
        source["buy_count"] = side.eq("buy").astype(int)
        source["sell_count"] = side.eq("sell").astype(int)
        source["_bin"] = source[timestamp_col].dt.ceil(frequency)
        grouped = source.groupby("_bin", sort=True)[list(columns)].sum()
        result = grouped.reindex(grid_index).fillna(0.0)
        result["coverage"] = 1.0
        return result

    aliases: Mapping[str, tuple[str, ...]] = {
        "buy_volume": ("buy_volume_1s", "buy_volume"),
        "sell_volume": ("sell_volume_1s", "sell_volume"),
        "buy_count": ("buy_count_1s", "buy_count"),
        "sell_count": ("sell_count_1s", "sell_count"),
    }
    for output, candidates in aliases.items():
        candidate = next((name for name in candidates if name in source), None)
        source[output] = pd.to_numeric(source[candidate], errors="coerce") if candidate else 0.0
    coverage_name = next(
        (name for name in ("trade_coverage_1s", "trade_coverage", "coverage") if name in source),
        None,
    )
    explicit_coverage = coverage_name is not None
    source["coverage"] = pd.to_numeric(source[coverage_name], errors="coerce") if coverage_name else 1.0
    if source[[*columns, "coverage"]].isna().any().any():
        raise ValueError("trade_seconds contains non-numeric aggregates or coverage")
    if (source[list(columns)] < 0).any().any() or (~source["coverage"].between(0, 1)).any():
        raise ValueError("trade aggregates must be non-negative and coverage must be in [0,1]")
    source["_bin"] = source[timestamp_col].dt.ceil(frequency)
    grouped = source.groupby("_bin", sort=True).agg(
        buy_volume=("buy_volume", "sum"),
        sell_volume=("sell_volume", "sum"),
        buy_count=("buy_count", "sum"),
        sell_count=("sell_count", "sum"),
        coverage=("coverage", "min"),
    )
    result = grouped.reindex(grid_index)
    if explicit_coverage:
        result["coverage"] = result["coverage"].fillna(0.0)
    else:
        result["coverage"] = 1.0
        result[list(columns)] = result[list(columns)].fillna(0.0)
    return result


def _ofi_base(
    ofi_seconds: pd.DataFrame | None,
    grid_index: pd.DatetimeIndex,
    *,
    timestamp_col: str,
    frequency: pd.Timedelta,
) -> pd.DataFrame:
    if ofi_seconds is None or ofi_seconds.empty:
        return pd.DataFrame({"ofi": np.nan, "coverage": 0.0}, index=grid_index)
    source = ofi_seconds.copy()
    if timestamp_col not in source:
        raise ValueError(f"ofi_seconds is missing {timestamp_col!r}")
    source[timestamp_col] = _utc(source[timestamp_col], timestamp_col)
    value_name = next(
        (name for name in ("ofi_event_sum_1s", "ofi_event", "ofi") if name in source),
        None,
    )
    if value_name is None:
        raise ValueError("ofi_seconds needs ofi_event_sum_1s, ofi_event, or ofi")
    source["ofi"] = pd.to_numeric(source[value_name], errors="coerce")
    if source["ofi"].isna().any():
        raise ValueError("OFI values must be finite")
    coverage_name = next(
        (name for name in ("ofi_coverage_1s", "ofi_coverage", "coverage") if name in source),
        None,
    )
    explicit_coverage = coverage_name is not None
    source["coverage"] = pd.to_numeric(source[coverage_name], errors="coerce") if coverage_name else 1.0
    if source["coverage"].isna().any() or (~source["coverage"].between(0, 1)).any():
        raise ValueError("OFI coverage must be in [0,1]")
    source["_bin"] = source[timestamp_col].dt.ceil(frequency)
    grouped = source.groupby("_bin", sort=True).agg(ofi=("ofi", "sum"), coverage=("coverage", "min"))
    result = grouped.reindex(grid_index)
    if explicit_coverage:
        result["coverage"] = result["coverage"].fillna(0.0)
    else:
        result["coverage"] = 1.0
        result["ofi"] = result["ofi"].fillna(0.0)
    return result


def build_feature_table(
    book_grid: pd.DataFrame,
    trade_seconds: pd.DataFrame | None = None,
    ofi_seconds: pd.DataFrame | None = None,
    *,
    timestamp_col: str = "timestamp",
    start: object | None = None,
    end: object | None = None,
    frequency: object = "1s",
    max_book_age_ms: float | None = None,
    depths: Iterable[int] = DEFAULT_DEPTHS,
    trade_windows_s: Iterable[object] = DEFAULT_TRADE_WINDOWS_S,
    ofi_windows_s: Iterable[object] = DEFAULT_OFI_WINDOWS_S,
    return_lags_s: Iterable[object] = DEFAULT_RETURN_LAGS_S,
    volatility_windows_s: Iterable[object] = DEFAULT_VOLATILITY_WINDOWS_S,
    numeric_epsilon: float = 1e-12,
) -> pd.DataFrame:
    """Build frozen-schema features using only locally available past data."""
    if timestamp_col not in book_grid:
        raise ValueError(f"book_grid is missing {timestamp_col!r}")
    if book_grid.empty and (start is None or end is None):
        raise ValueError("empty book_grid requires explicit start and end")

    step = _duration(frequency, "frequency")
    depths = tuple(sorted(set(int(value) for value in depths)))
    if not depths or min(depths) < 1:
        raise ValueError("depths must contain positive integers")
    max_depth = max(max(depths), 20)  # frozen processed schema always retains L1..L20

    states = _expand_level_lists(book_grid, max_depth)
    states[timestamp_col] = _utc(states[timestamp_col], timestamp_col)
    states["_input_order"] = np.arange(len(states))
    states = states.sort_values([timestamp_col, "_input_order"], kind="stable")
    states["_book_available_at"] = states[timestamp_col]

    inferred_start = states[timestamp_col].min() if start is None else _utc_timestamp(start, "start")
    inferred_end = states[timestamp_col].max() if end is None else _utc_timestamp(end, "end")
    grid_start = inferred_start.ceil(step)
    grid_end = inferred_end.floor(step)
    if grid_start > grid_end:
        raise ValueError("start/end contain no complete grid tick")
    grid_index = pd.date_range(grid_start, grid_end, freq=step, tz="UTC")
    grid = pd.DataFrame({timestamp_col: grid_index})
    result = pd.merge_asof(
        grid,
        states.drop(columns="_input_order"),
        on=timestamp_col,
        direction="backward",
        allow_exact_matches=True,
    )

    for side in ("bid", "ask"):
        for level in range(1, max_depth + 1):
            for field in ("price", "size"):
                name = _level_name(side, field, level)
                result[name] = pd.to_numeric(result[name], errors="coerce")

    if "book_recv_utc_ns" in result:
        receive_time = pd.to_datetime(
            pd.to_numeric(result["book_recv_utc_ns"], errors="coerce"), unit="ns", utc=True
        )
        calculated_age = (result[timestamp_col] - receive_time).dt.total_seconds() * 1_000
    else:
        calculated_age = (result[timestamp_col] - result["_book_available_at"]).dt.total_seconds() * 1_000
    if "book_age_ms" in result and result["book_age_ms"].notna().any():
        supplied_age = pd.to_numeric(result["book_age_ms"], errors="coerce")
        result["book_age_ms"] = supplied_age.where(supplied_age.notna(), calculated_age)
    else:
        result["book_age_ms"] = calculated_age

    base_valid = (
        result["book_valid"].astype("boolean").fillna(False).astype(bool)
        if "book_valid" in result
        else result["_book_available_at"].notna()
    )
    if "gap_or_resync_interval" in result:
        result["gap_or_resync_interval"] = result["gap_or_resync_interval"].astype("boolean").fillna(True).astype(bool)
        base_valid &= ~result["gap_or_resync_interval"]
    else:
        result["gap_or_resync_interval"] = result["_book_available_at"].isna()
    if max_book_age_ms is not None:
        if max_book_age_ms < 0:
            raise ValueError("max_book_age_ms must be non-negative")
        base_valid &= result["book_age_ms"].between(0, float(max_book_age_ms), inclusive="both")

    bid_price = result["bid_price_01"]
    bid_size = result["bid_size_01"]
    ask_price = result["ask_price_01"]
    ask_size = result["ask_size_01"]
    structural_valid = (
        np.isfinite(bid_price)
        & np.isfinite(ask_price)
        & np.isfinite(bid_size)
        & np.isfinite(ask_size)
        & (bid_price > 0)
        & (ask_price > 0)
        & (bid_size > 0)
        & (ask_size > 0)
        & (bid_price < ask_price)
    )
    result["book_valid"] = (base_valid & structural_valid).astype(bool)
    result["book_depth_bid"] = sum(
        (result[_level_name("bid", "price", level)] > 0)
        & (result[_level_name("bid", "size", level)] > 0)
        for level in range(1, max_depth + 1)
    )
    result["book_depth_ask"] = sum(
        (result[_level_name("ask", "price", level)] > 0)
        & (result[_level_name("ask", "size", level)] > 0)
        for level in range(1, max_depth + 1)
    )

    result["best_bid"] = bid_price.where(result["book_valid"])
    result["best_ask"] = ask_price.where(result["book_valid"])
    result["mid"] = ((result["best_bid"] + result["best_ask"]) / 2).where(result["book_valid"])
    result["spread_abs"] = (result["best_ask"] - result["best_bid"]).where(result["book_valid"])
    result["spread_bps"] = (10_000 * result["spread_abs"] / result["mid"]).where(
        result["mid"] > numeric_epsilon
    )
    l1_total = bid_size + ask_size
    result["microprice"] = ((ask_price * bid_size + bid_price * ask_size) / l1_total).where(
        result["book_valid"] & (l1_total > numeric_epsilon)
    )
    result["microdev"] = ((result["microprice"] - result["mid"]) / result["spread_abs"]).where(
        result["book_valid"] & (result["spread_abs"] > numeric_epsilon)
    )

    for depth in depths:
        bid_columns = [_level_name("bid", "size", level) for level in range(1, depth + 1)]
        ask_columns = [_level_name("ask", "size", level) for level in range(1, depth + 1)]
        bid_complete = result[bid_columns].gt(0).all(axis=1) & result[bid_columns].notna().all(axis=1)
        ask_complete = result[ask_columns].gt(0).all(axis=1) & result[ask_columns].notna().all(axis=1)
        bid_depth = result[bid_columns].sum(axis=1).where(result["book_valid"] & bid_complete)
        ask_depth = result[ask_columns].sum(axis=1).where(result["book_valid"] & ask_complete)
        result[f"bid_depth_{depth}"] = bid_depth
        result[f"ask_depth_{depth}"] = ask_depth
        total_depth = bid_depth + ask_depth
        result[f"obi_{depth}"] = ((bid_depth - ask_depth) / total_depth).where(
            total_depth > numeric_epsilon
        )

    # Invalid book rows retain quality metadata but never stale numerical state.
    level_columns = [
        _level_name(side, field, level)
        for side in ("bid", "ask")
        for field in ("price", "size")
        for level in range(1, max_depth + 1)
    ]
    result.loc[~result["book_valid"], level_columns] = np.nan
    result["usable"] = result["book_valid"] & result["mid"].notna()

    trade_base = _trade_base(
        trade_seconds, grid_index, timestamp_col=timestamp_col, frequency=step
    )
    for window in trade_windows_s:
        window_td = _duration(window, "trade window")
        label = _seconds_label(window)
        expected = int(np.ceil(window_td / step))
        coverage = trade_base["coverage"].rolling(
            window_td, closed="right", min_periods=expected
        ).mean()
        full = coverage.ge(1 - 1e-12)
        buy_volume = trade_base["buy_volume"].rolling(
            window_td, closed="right", min_periods=expected
        ).sum().where(full)
        sell_volume = trade_base["sell_volume"].rolling(
            window_td, closed="right", min_periods=expected
        ).sum().where(full)
        buy_count = trade_base["buy_count"].rolling(
            window_td, closed="right", min_periods=expected
        ).sum().where(full)
        sell_count = trade_base["sell_count"].rolling(
            window_td, closed="right", min_periods=expected
        ).sum().where(full)
        volume = buy_volume + sell_volume
        count = buy_count + sell_count
        result[f"buy_volume_{label}"] = buy_volume.to_numpy()
        result[f"sell_volume_{label}"] = sell_volume.to_numpy()
        result[f"signed_volume_{label}"] = (buy_volume - sell_volume).to_numpy()
        result[f"volume_imbalance_{label}"] = np.where(
            full, np.divide(buy_volume - sell_volume, volume, out=np.zeros(len(result)), where=volume.gt(0)), np.nan
        )
        result[f"buy_count_{label}"] = buy_count.to_numpy()
        result[f"sell_count_{label}"] = sell_count.to_numpy()
        result[f"trade_count_imbalance_{label}"] = np.where(
            full, np.divide(buy_count - sell_count, count, out=np.zeros(len(result)), where=count.gt(0)), np.nan
        )
        result[f"total_trade_volume_{label}"] = volume.to_numpy()
        result[f"trade_count_{label}"] = count.to_numpy()
        result[f"trade_coverage_{label}"] = coverage.to_numpy()

    ofi_base = _ofi_base(ofi_seconds, grid_index, timestamp_col=timestamp_col, frequency=step)
    result["ofi_event_sum_1s"] = ofi_base["ofi"].where(ofi_base["coverage"].ge(1 - 1e-12)).to_numpy()
    for window in ofi_windows_s:
        window_td = _duration(window, "OFI window")
        label = _seconds_label(window)
        expected = int(np.ceil(window_td / step))
        coverage = ofi_base["coverage"].rolling(
            window_td, closed="right", min_periods=expected
        ).mean()
        full = coverage.ge(1 - 1e-12)
        values = ofi_base["ofi"].rolling(
            window_td, closed="right", min_periods=expected
        ).sum().where(full)
        result[f"ofi_{label}"] = values.to_numpy()
        result[f"ofi_coverage_{label}"] = coverage.to_numpy()

    for lag in return_lags_s:
        lag_td = _duration(lag, "return lag")
        label = _seconds_label(lag)
        past_mid = _exact_lookup(result["mid"], result[timestamp_col], lag_td)
        current_mid = result["mid"].to_numpy(float)
        valid = (current_mid > 0) & (past_mid > 0)
        values = np.full(len(result), np.nan)
        values[valid] = np.log(current_mid[valid] / past_mid[valid])
        result[f"return_{label}"] = values

    one_step_label = _seconds_label(step)
    one_step_name = f"return_{one_step_label}"
    if one_step_name not in result:
        past_mid = _exact_lookup(result["mid"], result[timestamp_col], step)
        current_mid = result["mid"].to_numpy(float)
        values = np.full(len(result), np.nan)
        valid = (current_mid > 0) & (past_mid > 0)
        values[valid] = np.log(current_mid[valid] / past_mid[valid])
        result[one_step_name] = values
    squared_return = pd.Series(result[one_step_name].to_numpy() ** 2, index=grid_index)
    for window in volatility_windows_s:
        window_td = _duration(window, "volatility window")
        label = _seconds_label(window)
        expected = int(np.ceil(window_td / step))
        realized_variance = squared_return.rolling(
            window_td, closed="right", min_periods=expected
        ).sum()
        result[f"realized_vol_{label}"] = np.sqrt(realized_variance).to_numpy()

    result = result.drop(columns=["_book_available_at", "bids", "asks"], errors="ignore")
    return result[[timestamp_col, *[column for column in result if column != timestamp_col]]]


__all__ = ["build_feature_table", "compute_event_ofi", "ofi_event"]
