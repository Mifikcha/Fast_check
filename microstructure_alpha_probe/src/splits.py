"""Chronological split masks with common-horizon purge and embargo."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


def _duration(value: object, name: str) -> pd.Timedelta:
    result = pd.Timedelta(seconds=float(value)) if isinstance(value, (int, float, np.number)) else pd.Timedelta(value)
    if result < pd.Timedelta(0):
        raise ValueError(f"{name} must be non-negative")
    return result


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


def _timestamps(data: pd.DataFrame | pd.Series | pd.Index | Sequence[object], timestamp_col: str) -> pd.Series:
    if isinstance(data, pd.DataFrame):
        if timestamp_col not in data:
            raise ValueError(f"data is missing {timestamp_col!r}")
        raw = data[timestamp_col]
        index = data.index
    elif isinstance(data, pd.Series):
        raw = data
        index = data.index
    else:
        raw = pd.Series(data)
        index = raw.index
    try:
        converted = pd.to_datetime(raw, utc=True, errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{timestamp_col} must contain parseable UTC timestamps") from exc
    result = pd.Series(converted, index=index, name=timestamp_col)
    if result.empty:
        raise ValueError("cannot split an empty time grid")
    if result.isna().any() or result.duplicated().any():
        raise ValueError("split timestamps must be finite and unique")
    if not result.is_monotonic_increasing:
        raise ValueError("split timestamps must be chronologically sorted")
    return result


def _labels(index: pd.Index, **masks: pd.Series) -> pd.Series:
    result = pd.Series(pd.NA, index=index, dtype="string", name="split")
    for name, mask in masks.items():
        result.loc[mask] = name
    return result


def make_chronological_split(
    data: pd.DataFrame | pd.Series | pd.Index | Sequence[object],
    *,
    timestamp_col: str = "timestamp",
    train_fraction: float = 0.60,
    validation_fraction: float = 0.20,
    test_fraction: float = 0.20,
    max_horizon_s: object = 60,
    embargo_s: object | None = None,
    train_validation_boundary: object | None = None,
    validation_test_boundary: object | None = None,
) -> dict[str, object]:
    """Return auditable outer split masks on the complete, unfiltered grid.

    Source rows are purged when ``timestamp + max_horizon >= boundary``.
    Destination rows begin at ``boundary + embargo``. The same maximum horizon
    is therefore used for every target/model comparison.
    """
    times = _timestamps(data, timestamp_col)
    fractions = np.asarray([train_fraction, validation_fraction, test_fraction], dtype=float)
    if not np.isfinite(fractions).all() or (fractions <= 0).any() or not np.isclose(fractions.sum(), 1.0):
        raise ValueError("train/validation/test fractions must be positive and sum to one")
    if len(times) < 3:
        raise ValueError("at least three grid rows are required")

    if train_validation_boundary is None:
        first_cut = int(np.floor(len(times) * train_fraction))
        if not 0 < first_cut < len(times):
            raise ValueError("train fraction produces an invalid boundary")
        boundary_1 = times.iloc[first_cut]
    else:
        boundary_1 = _utc_timestamp(train_validation_boundary, "train_validation_boundary")
    if validation_test_boundary is None:
        second_cut = int(np.floor(len(times) * (train_fraction + validation_fraction)))
        if not 0 < second_cut < len(times):
            raise ValueError("validation fraction produces an invalid boundary")
        boundary_2 = times.iloc[second_cut]
    else:
        boundary_2 = _utc_timestamp(validation_test_boundary, "validation_test_boundary")
    if boundary_1 not in set(times) or boundary_2 not in set(times):
        raise ValueError("split boundaries must be exact timestamps on the complete grid")
    if boundary_1 >= boundary_2:
        raise ValueError("train/validation boundary must precede validation/test boundary")

    horizon = _duration(max_horizon_s, "max_horizon_s")
    embargo = horizon if embargo_s is None else _duration(embargo_s, "embargo_s")
    if embargo < horizon:
        raise ValueError("embargo must be at least the maximum prediction horizon")

    label_end = times + horizon
    train_raw = times.lt(boundary_1)
    validation_raw = times.ge(boundary_1) & times.lt(boundary_2)
    test_raw = times.ge(boundary_2)
    train = train_raw & label_end.lt(boundary_1)
    validation = (
        validation_raw
        & times.ge(boundary_1 + embargo)
        & label_end.lt(boundary_2)
    )
    test = test_raw & times.ge(boundary_2 + embargo)
    purged = (train_raw & ~label_end.lt(boundary_1)) | (
        validation_raw & times.ge(boundary_1 + embargo) & ~label_end.lt(boundary_2)
    )
    embargoed = (
        validation_raw & times.lt(boundary_1 + embargo)
    ) | (test_raw & times.lt(boundary_2 + embargo))
    if not train.any() or not validation.any() or not test.any():
        raise ValueError("purge/embargo leaves an empty outer split")
    if (train & validation).any() or (train & test).any() or (validation & test).any():
        raise AssertionError("temporal split masks overlap")

    split_labels = _labels(times.index, train=train, validation=validation, test=test)
    return {
        "train": train,
        "validation": validation,
        "test": test,
        "purged": purged,
        "embargoed": embargoed,
        "excluded": ~(train | validation | test),
        "labels": split_labels,
        "train_validation_boundary": boundary_1,
        "validation_test_boundary": boundary_2,
        "max_horizon": horizon,
        "embargo": embargo,
    }


def make_walk_forward_splits(
    data: pd.DataFrame | pd.Series | pd.Index | Sequence[object],
    *,
    timestamp_col: str = "timestamp",
    n_splits: int = 3,
    max_horizon_s: object = 60,
    embargo_s: object | None = None,
    initial_train_size: int | None = None,
    validation_size: int | None = None,
) -> list[dict[str, object]]:
    """Create expanding-window train/validation masks.

    Pass only the development portion of the complete grid. Fold assessment
    chunks are consecutive; each has a leading embargo and a purged tail so its
    labels remain strictly inside that chunk.
    """
    times = _timestamps(data, timestamp_col)
    if not isinstance(n_splits, int) or n_splits < 1:
        raise ValueError("n_splits must be a positive integer")
    horizon = _duration(max_horizon_s, "max_horizon_s")
    embargo = horizon if embargo_s is None else _duration(embargo_s, "embargo_s")
    if embargo < horizon:
        raise ValueError("embargo must be at least the maximum prediction horizon")

    if validation_size is None:
        validation_size = len(times) // (n_splits + 1)
    if not isinstance(validation_size, int) or validation_size < 1:
        raise ValueError("validation_size must be a positive row count")
    if initial_train_size is None:
        initial_train_size = len(times) - n_splits * validation_size
    if not isinstance(initial_train_size, int) or initial_train_size < 1:
        raise ValueError("initial_train_size must be a positive row count")
    if initial_train_size + n_splits * validation_size > len(times):
        raise ValueError("requested walk-forward folds exceed the supplied grid")

    if len(times) > 1:
        observed = times.diff().dropna()
        step = observed.min()
        if (observed != step).any():
            # Exact time masks remain safe, but the final chunk needs an
            # exclusive right boundary. Use the smallest observed grid step.
            step = observed.min()
    else:
        raise ValueError("walk-forward splitting needs at least two timestamps")

    folds: list[dict[str, object]] = []
    for fold in range(n_splits):
        boundary_position = initial_train_size + fold * validation_size
        validation_end_position = boundary_position + validation_size
        boundary = times.iloc[boundary_position]
        validation_end = (
            times.iloc[validation_end_position]
            if validation_end_position < len(times)
            else times.iloc[-1] + step
        )
        label_end = times + horizon
        train = times.lt(boundary) & label_end.lt(boundary)
        validation_raw = times.ge(boundary) & times.lt(validation_end)
        validation = (
            validation_raw
            & times.ge(boundary + embargo)
            & label_end.lt(validation_end)
        )
        purged = (times.lt(boundary) & ~label_end.lt(boundary)) | (
            validation_raw & times.ge(boundary + embargo) & ~label_end.lt(validation_end)
        )
        embargoed = validation_raw & times.lt(boundary + embargo)
        if not train.any() or not validation.any():
            raise ValueError(f"purge/embargo leaves fold {fold} empty")
        folds.append(
            {
                "fold": fold,
                "train": train,
                "validation": validation,
                "purged": purged,
                "embargoed": embargoed,
                "excluded": ~(train | validation),
                "labels": _labels(times.index, train=train, validation=validation),
                "assessment_boundary": boundary,
                "validation_end": validation_end,
                "max_horizon": horizon,
                "embargo": embargo,
            }
        )
    return folds


__all__ = ["make_chronological_split", "make_walk_forward_splits"]
