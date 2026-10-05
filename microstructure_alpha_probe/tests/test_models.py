from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from src.config import load_config
from src.evaluate import classification_metrics
from src.models import (
    Candidate,
    aligned_class_probabilities,
    build_classifier,
    build_regressor,
    resolve_feature_groups,
)


def test_feature_allowlist_excludes_prices_targets_and_future() -> None:
    frame = pd.DataFrame(
        {
            "mid": [100.0, 101.0],
            "target_return_10s": [0.0, 0.1],
            "future_mid_10s": [101.0, 102.0],
            "return_1s": [0.0, 0.01],
            "obi_5": [0.1, 0.2],
            "volume_imbalance_5s": [-0.2, 0.3],
        }
    )
    groups = resolve_feature_groups(frame)
    assert groups["price_history"] == ["return_1s"]
    assert groups["order_book"] == ["obi_5"]
    assert groups["trade_flow"] == ["volume_imbalance_5s"]
    assert not any("target" in name or "future" in name for name in groups["all"])


def test_train_only_pipeline_and_probability_label_alignment() -> None:
    x = pd.DataFrame({"return_1s": [-2.0, -1.0, 0.0, 1.0, 2.0, np.nan]})
    y = np.array([-1, -1, 0, 1, 1, 0])
    candidate = Candidate("logistic_c1", "logistic", {"C": 1.0})
    model = build_classifier(candidate, seed=42, n_jobs=1).fit(x, y)
    probabilities = aligned_class_probabilities(model, x)
    assert probabilities.shape == (6, 3)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)

    regression = build_regressor(Candidate("ridge_a1", "ridge", {"alpha": 1.0}), seed=42, n_jobs=1)
    pred = regression.fit(x.iloc[:5], np.arange(5.0)).predict(x.iloc[5:])
    assert pred.shape == (1,)
    assert regression.named_steps["imputer"].statistics_[0] == 0.0  # train median, not future NaN fill


def test_one_class_classification_metrics_are_quiet_and_explicitly_undefined() -> None:
    probabilities = np.array([[0.1, 0.8, 0.1], [0.1, 0.2, 0.7]])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        metrics, confusion, _ = classification_metrics([0, 0], [0, 1], probabilities)
    assert metrics["balanced_accuracy"] == 0.5
    assert np.isnan(metrics["mcc"])
    assert np.isnan(metrics["roc_auc_ovr_macro"])
    assert np.isnan(metrics["pr_auc_macro"])
    assert len(confusion) == 9
