import numpy as np
import pandas as pd

from src.features import build_feature_table, compute_event_ofi, ofi_event


def _time(seconds: int | float) -> pd.Timestamp:
    return pd.Timestamp("2026-01-01T00:00:00Z") + pd.Timedelta(seconds=seconds)


def _book_row(second: int, bid: float, bid_size: float, ask: float, ask_size: float) -> dict:
    return {
        "timestamp": _time(second),
        "session_id": "s",
        "connection_id": "c",
        "symbol": "BTCUSDT",
        "book_valid": True,
        "type": "snapshot" if second == 0 else "delta",
        "bids": [(bid, bid_size), (bid - 1, 2.0)],
        "asks": [(ask, ask_size), (ask + 1, 2.0)],
    }


def test_price_book_trade_ofi_and_return_features_have_known_values():
    books = pd.DataFrame(
        [
            _book_row(0, 99, 3, 101, 1),
            _book_row(1, 100, 2, 102, 2),
            _book_row(2, 102, 1, 104, 3),
        ]
    )
    trades = pd.DataFrame(
        {
            "timestamp": [_time(0), _time(1), _time(2)],
            "buy_volume_1s": [9.0, 2.0, 0.0],
            "sell_volume_1s": [1.0, 0.0, 3.0],
            "buy_count_1s": [2, 1, 0],
            "sell_count_1s": [1, 0, 1],
            "trade_coverage_1s": [1.0, 1.0, 1.0],
        }
    )
    ofi = pd.DataFrame(
        {
            "timestamp": [_time(0), _time(1), _time(2)],
            "ofi_event_sum_1s": [100.0, 1.0, 2.0],
            "ofi_coverage_1s": [1.0, 1.0, 1.0],
        }
    )

    result = build_feature_table(
        books,
        trades,
        ofi,
        depths=(1, 2),
        trade_windows_s=(1, 2),
        ofi_windows_s=(1, 2),
        return_lags_s=(1, 2),
        volatility_windows_s=(2,),
    )

    assert str(result["timestamp"].dtype) == "datetime64[ns, UTC]"
    assert result["timestamp"].tolist() == [_time(0), _time(1), _time(2)]
    assert np.isclose(result.loc[0, "mid"], 100.0)
    assert np.isclose(result.loc[0, "microprice"], 100.5)
    assert np.isclose(result.loc[0, "microdev"], 0.25)
    assert np.isclose(result.loc[0, "bid_depth_1"], 3.0)
    assert np.isclose(result.loc[0, "ask_depth_1"], 1.0)
    assert np.isclose(result.loc[0, "obi_1"], 0.5)
    assert np.isclose(result.loc[2, "buy_volume_2s"], 2.0)
    assert np.isclose(result.loc[2, "sell_volume_2s"], 3.0)
    assert np.isclose(result.loc[2, "signed_volume_2s"], -1.0)
    assert np.isclose(result.loc[2, "volume_imbalance_2s"], -0.2)
    assert np.isclose(result.loc[2, "ofi_2s"], 3.0)  # t=0 is excluded from (0,2]
    assert np.isclose(result.loc[2, "return_2s"], np.log(103 / 100))
    expected_rv = np.sqrt(np.log(101 / 100) ** 2 + np.log(103 / 101) ** 2)
    assert np.isclose(result.loc[2, "realized_vol_2s"], expected_rv)


def test_sparse_book_is_backward_asof_by_local_availability_and_goes_stale():
    books = pd.DataFrame(
        [
            {**_book_row(0, 99, 1, 101, 1), "exchange_ts_ms": 9_999_999_999_999},
            {**_book_row(3, 199, 1, 201, 1), "exchange_ts_ms": 0},
        ]
    )
    result = build_feature_table(
        books,
        start=_time(0),
        end=_time(3),
        max_book_age_ms=1_500,
        depths=(1,),
        trade_windows_s=(1,),
        ofi_windows_s=(1,),
        return_lags_s=(1,),
        volatility_windows_s=(1,),
    )

    assert result["mid"].tolist()[:2] == [100.0, 100.0]
    assert result.loc[2, "book_valid"] == False  # noqa: E712
    assert np.isnan(result.loc[2, "mid"])
    assert result.loc[3, "mid"] == 200.0
    assert result["gap_or_resync_interval"].dtype == bool
    assert not result["gap_or_resync_interval"].isna().any()
    # The future event carried an old exchange timestamp but was unavailable
    # locally until t=3, so it cannot alter t=0..2.
    assert result.loc[1, "mid"] != 200.0


def test_appending_future_events_does_not_change_past_features():
    prefix = pd.DataFrame(
        [_book_row(0, 99, 1, 101, 1), _book_row(1, 100, 1, 102, 1), _book_row(2, 101, 1, 103, 1)]
    )
    future = pd.concat(
        [prefix, pd.DataFrame([_book_row(3, 999, 100, 1001, 1)])], ignore_index=True
    )
    kwargs = dict(
        start=_time(0),
        end=_time(2),
        depths=(1, 2),
        trade_windows_s=(1, 2),
        ofi_windows_s=(1, 2),
        return_lags_s=(1, 2),
        volatility_windows_s=(2,),
    )
    before = build_feature_table(prefix, **kwargs)
    after = build_feature_table(future, **kwargs)
    pd.testing.assert_frame_equal(before, after)


def test_ofi_scalar_cases_match_documented_signs():
    previous = (100.0, 2.0, 101.0, 3.0)
    assert ofi_event(*previous, 100.0, 5.0, 101.0, 1.0) == 5.0
    assert ofi_event(*previous, 100.5, 4.0, 101.0, 3.0) == 4.0
    assert ofi_event(*previous, 99.5, 7.0, 101.0, 3.0) == -2.0
    assert ofi_event(*previous, 100.0, 2.0, 100.5, 4.0) == -4.0
    assert ofi_event(*previous, 100.0, 2.0, 101.5, 6.0) == 3.0


def test_event_ofi_does_not_bridge_connections_or_score_snapshot():
    events = pd.DataFrame(
        [
            _book_row(0, 100, 2, 101, 3),
            _book_row(1, 100, 5, 101, 1),
            {**_book_row(2, 200, 1, 201, 1), "connection_id": "new", "type": "snapshot"},
            {**_book_row(3, 201, 4, 202, 1), "connection_id": "new", "type": "delta"},
        ]
    )
    # compute_event_ofi consumes frozen flattened L1 columns.
    for side, source in (("bid", "bids"), ("ask", "asks")):
        events[f"{side}_price_01"] = events[source].map(lambda levels: levels[0][0])
        events[f"{side}_size_01"] = events[source].map(lambda levels: levels[0][1])
    result = compute_event_ofi(events)
    assert np.isnan(result.iloc[0])
    assert result.iloc[1] == 5.0
    assert np.isnan(result.iloc[2])
    assert result.iloc[3] == 5.0  # improved bid +4 and worsened ask contributes +1


def test_incomplete_requested_depth_is_nan_not_variable_depth():
    books = pd.DataFrame([_book_row(0, 99, 1, 101, 1)])
    books.at[0, "asks"] = [(101, 1)]
    result = build_feature_table(
        books,
        depths=(1, 2),
        trade_windows_s=(1,),
        ofi_windows_s=(1,),
        return_lags_s=(1,),
        volatility_windows_s=(1,),
    )
    assert np.isnan(result.loc[0, "ask_depth_2"])
    assert np.isnan(result.loc[0, "obi_2"])
