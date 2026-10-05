import numpy as np
import pandas as pd

from src.targets import add_targets, directional_class


def _grid(seconds=(0, 1, 2, 3, 4)) -> pd.DataFrame:
    base = pd.Timestamp("2026-01-01T00:00:00Z")
    mid_by_second = {second: 100.0 + second for second in seconds}
    return pd.DataFrame(
        {
            "timestamp": [base + pd.Timedelta(seconds=second) for second in seconds],
            "mid": [mid_by_second[second] for second in seconds],
            "best_bid": [mid_by_second[second] - 0.5 for second in seconds],
            "best_ask": [mid_by_second[second] + 0.5 for second in seconds],
            "book_valid": True,
            "gap_or_resync_interval": False,
        }
    )


def test_exact_target_values_and_end_of_grid_nans():
    result = add_targets(_grid(), horizons_s=(2,), dead_zone_bps=0)
    expected = [np.log(102 / 100), np.log(103 / 101), np.log(104 / 102)]
    np.testing.assert_allclose(result.loc[:2, "target_return_2s"], expected)
    assert result.loc[:2, "target_valid_2s"].all()
    assert result.loc[:2, "target_path_valid_2s"].all()
    assert result.loc[3:, "target_return_2s"].isna().all()
    assert not result.loc[3:, "target_valid_2s"].any()
    assert result.loc[0, "target_end_time_2s"] == result.loc[2, "timestamp"]
    assert result.loc[0, "future_bid_2s"] == result.loc[2, "best_bid"]
    assert result.loc[0, "future_ask_2s"] == result.loc[2, "best_ask"]


def test_endpoint_and_path_validity_are_separate():
    grid = _grid()
    grid.loc[2, "book_valid"] = False
    result = add_targets(grid, horizons_s=(2,))
    assert not result.loc[0, "target_valid_2s"]  # invalid future endpoint
    assert result.loc[1, "target_valid_2s"]  # endpoints 1 and 3 are valid
    assert not result.loc[1, "target_path_valid_2s"]  # path crosses invalid t=2
    assert not result.loc[2, "target_valid_2s"]  # invalid current endpoint


def test_missing_physical_row_is_never_replaced_by_next_retained_row():
    grid = _grid(seconds=(0, 1, 3, 4))
    result = add_targets(grid, horizons_s=(2,))
    assert not result.loc[0, "target_valid_2s"]  # exact t=2 is absent
    assert np.isnan(result.loc[0, "target_return_2s"])
    assert result.loc[1, "target_valid_2s"]  # exact t=3 exists
    assert not result.loc[1, "target_path_valid_2s"]  # but complete path is unprovable
    assert np.isclose(result.loc[1, "target_return_2s"], np.log(103 / 101))


def test_dead_zone_boundaries_are_flat_and_invalid_stays_missing():
    returns = pd.Series([-0.0011, -0.001, 0.0, 0.001, 0.0011, np.nan])
    actual = directional_class(returns, 0.001)
    expected = pd.Series([-1, 0, 0, 0, 1, pd.NA], dtype="Int8")
    pd.testing.assert_series_equal(actual, expected)


def test_per_horizon_dead_zone_mapping_and_utc_timestamp():
    result = add_targets(
        _grid(), horizons_s=(1, 2), dead_zone_bps={1: 200, "2s": 0}
    )
    assert str(result["timestamp"].dtype) == "datetime64[ns, UTC]"
    assert result.loc[0, "target_class_1s"] == 0
    assert result.loc[0, "target_class_2s"] == 1
