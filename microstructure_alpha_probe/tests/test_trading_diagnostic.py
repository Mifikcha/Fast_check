from __future__ import annotations

import math

import numpy as np
import pandas as pd

from src.trading_diagnostic import build_non_overlapping_trades, trading_metrics


def _market(prediction: float, future_mid: float, future_bid: float, future_ask: float) -> pd.DataFrame:
    start = pd.Timestamp("2026-01-01T00:00:00Z")
    return pd.DataFrame({
        "timestamp": [start], "prediction": [prediction], "mid": [100.0],
        "best_bid": [99.0], "best_ask": [101.0], "book_valid": [True],
        "target_end_time_10s": [start + pd.Timedelta(seconds=10)],
        "future_mid_10s": [future_mid], "future_bid_10s": [future_bid],
        "future_ask_10s": [future_ask], "target_valid_10s": [True],
        "target_path_valid_10s": [True],
    })


def test_long_and_short_taker_pnl_and_incremental_cost() -> None:
    long = build_non_overlapping_trades(
        _market(0.01, 110.0, 109.0, 111.0), prediction_column="prediction",
        horizon_s=10, threshold=0, incremental_cost_bps=5,
    )
    assert np.isclose(long.loc[0, "gross_taker_log"], math.log(109 / 101))
    assert np.isclose(long.loc[0, "net_taker_log"], math.log(109 / 101) - 0.0005)

    short = build_non_overlapping_trades(
        _market(-0.01, 90.0, 89.0, 91.0), prediction_column="prediction",
        horizon_s=10, threshold=0, incremental_cost_bps=5,
    )
    assert np.isclose(short.loc[0, "gross_taker_log"], math.log(99 / 91))
    assert trading_metrics(short, execution="taker")["number_of_trades"] == 1


def test_non_overlapping_positions_reserve_until_exact_exit() -> None:
    start = pd.Timestamp("2026-01-01T00:00:00Z")
    frame = pd.concat([
        _market(0.01, 101.0, 100.9, 101.1),
        _market(0.01, 101.0, 100.9, 101.1).assign(timestamp=start + pd.Timedelta(seconds=5)),
    ], ignore_index=True)
    frame.loc[1, "target_end_time_10s"] = start + pd.Timedelta(seconds=15)
    trades = build_non_overlapping_trades(frame, prediction_column="prediction", horizon_s=10, threshold=0)
    assert len(trades) == 1
    assert trades.attrs["skipped_overlap_count"] == 1
