from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd

from src.config import load_config
from src.evaluate import analyze_dataset
from src.features import build_feature_table
from src.report import PLOT_NAMES, render_report
from src.targets import add_targets


def test_synthetic_full_analysis_writes_a_smoke_only_audit_bundle(tmp_path) -> None:
    config = deepcopy(load_config())
    config["paths"] = {name: str(tmp_path / name.removesuffix("_dir")) for name in ("raw_dir", "processed_dir", "runs_dir")}
    config["targets"].update(horizons_s=[2, 4, 6], dead_zone_bps=2.0)
    config["splits"].update(embargo_s=6, purge_s=6, walk_forward_folds=2, minimum_fold_train_rows=20)
    config["models"].update(
        primary_horizon_s=4, ridge_alphas=[0.1, 1.0],
        lightgbm_grid=[{"n_estimators": 8, "learning_rate": 0.1, "num_leaves": 7, "max_depth": 3, "min_child_samples": 10}],
        permutation_repeats=2, lightgbm_n_jobs=1,
    )
    config["features"].update(
        depth_levels=[1, 5, 10, 20], trade_windows_s=[1, 3, 6], ofi_windows_s=[1, 3, 6],
        return_lags_s=[1, 3, 6], volatility_windows_s=[3, 6, 12],
        important_features=["microdev", "obi_5", "volume_imbalance_3s", "ofi_3s", "return_3s"],
    )
    config["regimes"].update(volatility_feature="realized_vol_12s")
    config["statistics"].update(autocorrelation_lags_s=[1, 3, 6], minimum_group_n=20, max_mutual_information_rows=500)
    config["bootstrap"].update(n_resamples=10, block_multipliers=[1])
    config["quantiles"].update(bins=5, minimum_bucket_n=2)
    config["trading"].update(threshold_grid_bps=[0.0, 1.0], minimum_validation_trades=1,
                              incremental_round_trip_cost_bps=[0.0, 5.0])
    config["plots"]["dpi"] = 50

    n = 360
    timestamp = pd.date_range("2026-01-01", periods=n, freq="1s", tz="UTC")
    increments = 0.04 * np.sin(np.arange(n) / 4) + 0.025 * np.cos(np.arange(n) / 11)
    mid = 100 + np.cumsum(increments)
    states = pd.DataFrame({
        "timestamp": timestamp, "session_id": "synthetic", "connection_id": "one", "symbol": "BTCUSDT",
        "book_valid": True, "gap_or_resync_interval": False,
        "book_recv_utc_ns": timestamp.astype("int64"), "book_exchange_ts_ms": timestamp.astype("int64") // 1_000_000,
        "update_id": np.arange(n), "cross_seq": np.arange(n),
    })
    for level in range(1, 21):
        states[f"bid_price_{level:02d}"] = mid - 0.5 - 0.1 * (level - 1)
        states[f"ask_price_{level:02d}"] = mid + 0.5 + 0.1 * (level - 1)
        states[f"bid_size_{level:02d}"] = 2 + level / 10 + 0.3 * np.sin(np.arange(n) / 7)
        states[f"ask_size_{level:02d}"] = 2 + level / 10 - 0.3 * np.sin(np.arange(n) / 7)
    trades = pd.DataFrame({
        "timestamp": timestamp, "buy_volume": 2 + np.sin(np.arange(n) / 5),
        "sell_volume": 2 - np.sin(np.arange(n) / 5), "buy_count": 2, "sell_count": 2, "coverage": 1.0,
    })
    ofi = pd.DataFrame({"timestamp": timestamp, "ofi_event": np.sin(np.arange(n) / 7), "coverage": 1.0})
    research = build_feature_table(
        states, trades, ofi, depths=config["features"]["depth_levels"],
        trade_windows_s=config["features"]["trade_windows_s"], ofi_windows_s=config["features"]["ofi_windows_s"],
        return_lags_s=config["features"]["return_lags_s"], volatility_windows_s=config["features"]["volatility_windows_s"],
        max_book_age_ms=2_000,
    )
    research = add_targets(research, horizons_s=[2, 4, 6], dead_zone_bps=2.0)
    dataset = tmp_path / "processed" / "synthetic" / "research_table.parquet"
    dataset.parent.mkdir(parents=True)
    research.to_parquet(dataset, index=False)

    run, verdict = analyze_dataset(config, dataset, capture="synthetic")
    assert verdict["verdict"] == "NO-GO — INSUFFICIENT EVIDENCE (SMOKE ONLY)"
    assert {"INSUFFICIENT_DURATION", "INSUFFICIENT_TEST_ROWS"} <= set(verdict["reason_codes"])
    assert (run / "decision_lock.json").exists()
    assert (run / "predictions_test.parquet").exists()
    assert (run / "data_quality_report.md").exists()
    assert all((run / "plots" / name).exists() for name in PLOT_NAMES)
    first = (run / "report.md").read_text(encoding="utf-8")
    assert render_report(run).read_text(encoding="utf-8") == first
