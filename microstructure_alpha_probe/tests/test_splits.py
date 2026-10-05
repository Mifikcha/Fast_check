import pandas as pd
import pytest

from src.splits import make_chronological_split, make_walk_forward_splits


def _grid(n: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=n, freq="1s", tz="UTC"),
            # Split logic must ignore post-quality filtering columns.
            "usable": [(i % 2) == 0 for i in range(n)],
        }
    )


def _seconds(mask: pd.Series, grid: pd.DataFrame) -> list[int]:
    start = grid["timestamp"].iloc[0]
    return ((grid.loc[mask, "timestamp"] - start).dt.total_seconds().astype(int)).tolist()


def test_outer_split_uses_strict_common_purge_and_destination_embargo():
    grid = _grid(30)
    split = make_chronological_split(
        grid,
        max_horizon_s=3,
        embargo_s=3,
        train_validation_boundary=grid.loc[10, "timestamp"],
        validation_test_boundary=grid.loc[20, "timestamp"],
    )
    assert _seconds(split["train"], grid) == list(range(0, 7))
    assert _seconds(split["validation"], grid) == list(range(13, 17))
    assert _seconds(split["test"], grid) == list(range(23, 30))
    assert (grid.loc[split["train"], "timestamp"] + pd.Timedelta(seconds=3)).max() < grid.loc[10, "timestamp"]
    assert (grid.loc[split["validation"], "timestamp"] + pd.Timedelta(seconds=3)).max() < grid.loc[20, "timestamp"]
    assert not (split["train"] & split["validation"]).any()
    assert not (split["train"] & split["test"]).any()
    assert not (split["validation"] & split["test"]).any()
    assert split["labels"].loc[split["train"]].eq("train").all()


def test_fraction_boundaries_are_chosen_before_usable_filtering():
    grid = _grid(100)
    split = make_chronological_split(grid, max_horizon_s=2, embargo_s=2)
    assert split["train_validation_boundary"] == grid.loc[60, "timestamp"]
    assert split["validation_test_boundary"] == grid.loc[80, "timestamp"]
    changed_quality = grid.copy()
    changed_quality["usable"] = False
    second = make_chronological_split(changed_quality, max_horizon_s=2, embargo_s=2)
    pd.testing.assert_series_equal(split["train"], second["train"])
    pd.testing.assert_series_equal(split["validation"], second["validation"])
    pd.testing.assert_series_equal(split["test"], second["test"])


def test_walk_forward_folds_expand_and_keep_labels_inside_assessment_chunk():
    grid = _grid(60)
    folds = make_walk_forward_splits(
        grid,
        n_splits=2,
        initial_train_size=20,
        validation_size=20,
        max_horizon_s=2,
        embargo_s=2,
    )
    assert _seconds(folds[0]["train"], grid) == list(range(0, 18))
    assert _seconds(folds[0]["validation"], grid) == list(range(22, 38))
    assert _seconds(folds[1]["train"], grid) == list(range(0, 38))
    assert _seconds(folds[1]["validation"], grid) == list(range(42, 58))
    assert folds[0]["assessment_boundary"] == grid.loc[20, "timestamp"]
    assert folds[1]["assessment_boundary"] == grid.loc[40, "timestamp"]


def test_rejects_short_embargo_unsorted_duplicate_and_empty_masks():
    with pytest.raises(ValueError, match="embargo"):
        make_chronological_split(_grid(30), max_horizon_s=3, embargo_s=2)
    with pytest.raises(ValueError, match="sorted"):
        make_chronological_split(_grid(30).iloc[::-1], max_horizon_s=1)
    duplicate = _grid(30)
    duplicate.loc[1, "timestamp"] = duplicate.loc[0, "timestamp"]
    with pytest.raises(ValueError, match="unique"):
        make_chronological_split(duplicate, max_horizon_s=1)
    with pytest.raises(ValueError, match="empty"):
        make_chronological_split(_grid(10), max_horizon_s=4, embargo_s=4)
