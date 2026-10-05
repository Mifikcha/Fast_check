from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class Candidate:
    name: str
    family: str
    params: dict[str, Any]


def _is_feature(name: str) -> bool:
    forbidden = (
        "target_", "future_", "target_end_", "split_", "prediction_",
        "book_exchange_", "book_recv_", "ticker_bid1", "ticker_ask1",
    )
    if name.startswith(forbidden):
        return False
    exact_forbidden = {
        "timestamp", "session_id", "symbol", "book_valid", "usable",
        "gap_or_resync_interval", "update_id", "cross_seq", "book_age_ms",
        "ticker_age_ms", "book_depth_bid", "book_depth_ask", "best_bid",
        "best_ask", "mid", "microprice", "spread_abs", "open_interest",
        "next_funding_time_ms",
    }
    return name not in exact_forbidden and not (
        name.startswith("bid_price_") or name.startswith("ask_price_")
        or name.startswith("bid_size_") or name.startswith("ask_size_")
    )


def resolve_feature_groups(frame: pd.DataFrame) -> dict[str, list[str]]:
    numeric = [c for c in frame.columns if pd.api.types.is_numeric_dtype(frame[c]) and _is_feature(c)]
    price = [c for c in numeric if c.startswith("return_") or c.startswith("realized_vol_")]
    order_book = [
        c for c in numeric
        if c in {"spread_bps", "microdev"}
        or c.startswith(("obi_", "bid_depth_", "ask_depth_", "ofi_"))
    ]
    trade = [
        c for c in numeric
        if c.startswith((
            "buy_volume_", "sell_volume_", "signed_volume_", "volume_imbalance_",
            "buy_count_", "sell_count_", "trade_count_", "total_trade_volume_",
        ))
    ]
    auxiliary = [
        c for c in numeric
        if c in {"funding_rate", "open_interest_delta_60s", "time_to_funding_s"}
    ]
    groups = {
        "price_history": price,
        "order_book": order_book,
        "trade_flow": trade,
        "order_book_trade_flow": sorted(set(order_book + trade)),
        "all": sorted(set(price + order_book + trade + auxiliary)),
    }
    if not groups["all"]:
        raise ValueError("No model features were found in the processed table")
    return groups


def regression_candidates(config: dict[str, Any]) -> list[Candidate]:
    model_cfg = config["models"]
    candidates = [
        Candidate("zero", "naive", {}),
        Candidate("historical_mean", "naive", {}),
        Candidate("previous_return", "naive", {}),
        Candidate("ols", "linear", {}),
    ]
    candidates.extend(
        Candidate(f"ridge_a{float(alpha):g}", "ridge", {"alpha": float(alpha)})
        for alpha in model_cfg["ridge_alphas"]
    )
    candidates.extend(
        Candidate(f"lightgbm_{index + 1}", "lightgbm", dict(params))
        for index, params in enumerate(model_cfg["lightgbm_grid"])
    )
    return candidates


def classification_candidates(config: dict[str, Any]) -> list[Candidate]:
    return [Candidate("majority_class", "naive", {})] + [
        Candidate(f"logistic_c{float(c):g}", "logistic", {"C": float(c)})
        for c in config["models"]["logistic_c"]
    ]


def build_regressor(candidate: Candidate, *, seed: int, n_jobs: int = -1) -> Pipeline:
    if candidate.family == "linear":
        estimator: Any = LinearRegression(n_jobs=n_jobs)
    elif candidate.family == "ridge":
        estimator = Ridge(alpha=float(candidate.params["alpha"]))
    elif candidate.family == "lightgbm":
        estimator = LGBMRegressor(
            objective="regression",
            random_state=seed,
            n_jobs=n_jobs,
            verbosity=-1,
            importance_type="gain",
            **candidate.params,
        )
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("model", estimator),
        ])
    else:
        raise ValueError(f"Cannot build regressor for {candidate.family}")
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scaler", StandardScaler()),
        ("model", estimator),
    ])


def build_classifier(candidate: Candidate, *, seed: int, n_jobs: int = -1) -> Pipeline:
    if candidate.family != "logistic":
        raise ValueError(f"Cannot build classifier for {candidate.family}")
    estimator = LogisticRegression(
        C=float(candidate.params["C"]),
        class_weight="balanced",
        max_iter=2000,
        random_state=seed,
    )
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scaler", StandardScaler()),
        ("model", estimator),
    ])


def naive_regression_prediction(
    candidate: Candidate,
    y_train: np.ndarray,
    evaluation_frame: pd.DataFrame,
) -> np.ndarray:
    if candidate.name == "zero":
        return np.zeros(len(evaluation_frame), dtype=float)
    if candidate.name == "historical_mean":
        return np.full(len(evaluation_frame), float(np.nanmean(y_train)), dtype=float)
    if candidate.name == "previous_return":
        if "return_1s" not in evaluation_frame:
            return np.zeros(len(evaluation_frame), dtype=float)
        return evaluation_frame["return_1s"].fillna(0.0).to_numpy(dtype=float)
    raise ValueError(f"Unknown naive regression candidate: {candidate.name}")


def majority_class_prediction(y_train: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray]:
    labels = np.array([-1, 0, 1], dtype=int)
    counts = np.array([(np.asarray(y_train) == label).sum() for label in labels], dtype=float)
    majority = int(labels[int(np.argmax(counts))])
    prediction = np.full(size, majority, dtype=int)
    probabilities = np.tile(counts / max(counts.sum(), 1.0), (size, 1))
    return prediction, probabilities


def aligned_class_probabilities(model: Pipeline, x: pd.DataFrame) -> np.ndarray:
    raw = model.predict_proba(x)
    classes = np.asarray(model.named_steps["model"].classes_)
    output = np.zeros((len(x), 3), dtype=float)
    for source, label in enumerate(classes):
        output[:, int(label) + 1] = raw[:, source]
    return output


def standardized_coefficients(model: Pipeline, feature_names: list[str]) -> pd.DataFrame:
    estimator = model.named_steps["model"]
    if not hasattr(estimator, "coef_"):
        return pd.DataFrame(columns=["feature", "coefficient"])
    coef = np.asarray(estimator.coef_, dtype=float)
    if coef.ndim == 2:
        coef = np.mean(np.abs(coef), axis=0)
    if len(coef) != len(feature_names):
        return pd.DataFrame(columns=["feature", "coefficient"])
    return pd.DataFrame({"feature": feature_names, "coefficient": coef})


def builtin_importance(model: Pipeline, feature_names: list[str]) -> pd.DataFrame:
    estimator = model.named_steps["model"]
    values = getattr(estimator, "feature_importances_", None)
    if values is None or len(values) != len(feature_names):
        return pd.DataFrame(columns=["feature", "importance"])
    return pd.DataFrame({"feature": feature_names, "importance": np.asarray(values, dtype=float)})
