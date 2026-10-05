from __future__ import annotations

import numpy as np

from src.statistics import benjamini_hochberg, fit_quantile_edges, moving_block_indices


def test_bh_preserves_nan_and_controls_declared_family() -> None:
    q, reject = benjamini_hochberg([0.01, 0.04, np.nan, 0.20], alpha=0.05)
    assert np.isnan(q[2])
    assert reject.tolist() == [True, False, False, False]


def test_blocks_never_cross_invalid_observation() -> None:
    indices = moving_block_indices(
        10, 3, rng=np.random.default_rng(1),
        valid_mask=[True, True, True, False, True, True, True, False, True, True],
        sample_size=6,
    )
    assert not np.isin(indices, [3, 7]).any()
    assert np.all(np.diff(indices.reshape(-1, 3), axis=1) == 1)


def test_tied_quantiles_collapse_instead_of_splitting_equal_values() -> None:
    edges = fit_quantile_edges([0, 0, 0, 1, 1, 1], requested_buckets=10)
    assert len(edges) < 11
