from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "pilot.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = Path(path or DEFAULT_CONFIG).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a mapping: {config_path}")
    validate_config(config)
    config = deepcopy(config)
    config["_config_path"] = str(config_path)
    return config


def validate_config(config: dict[str, Any]) -> None:
    required = {
        "project", "paths", "collection", "sampling", "features", "targets",
        "data_quality", "splits", "models", "statistics", "bootstrap",
        "quantiles", "trading", "regimes", "robustness", "verdict", "plots",
    }
    missing = required - config.keys()
    if missing:
        raise ValueError(f"Missing configuration sections: {sorted(missing)}")

    symbol = str(config["project"].get("symbol", "")).strip()
    if not symbol or symbol != symbol.upper():
        raise ValueError("project.symbol must be a non-empty uppercase Bybit symbol")

    interval = int(config["sampling"]["interval_s"])
    horizons = [int(v) for v in config["targets"]["horizons_s"]]
    if interval <= 0 or not horizons or any(h <= 0 or h % interval for h in horizons):
        raise ValueError("Every positive target horizon must be divisible by sampling.interval_s")

    fractions = [
        float(config["splits"]["train_fraction"]),
        float(config["splits"]["validation_fraction"]),
        float(config["splits"]["test_fraction"]),
    ]
    if any(v <= 0 for v in fractions) or abs(sum(fractions) - 1.0) > 1e-9:
        raise ValueError("Train/validation/test fractions must be positive and sum to one")

    max_horizon = max(horizons)
    if int(config["splits"]["embargo_s"]) < max_horizon:
        raise ValueError("splits.embargo_s must be at least the maximum horizon")
    if int(config["splits"]["purge_s"]) < max_horizon:
        raise ValueError("splits.purge_s must be at least the maximum horizon")

    costs = [float(v) for v in config["trading"]["incremental_round_trip_cost_bps"]]
    if any(v < 0 for v in costs):
        raise ValueError("Trading cost scenarios cannot be negative")


def project_path(config: dict[str, Any], key: str) -> Path:
    value = Path(config["paths"][key])
    return value if value.is_absolute() else PROJECT_ROOT / value


def public_config(config: dict[str, Any]) -> dict[str, Any]:
    return {k: deepcopy(v) for k, v in config.items() if not k.startswith("_")}


def dump_config(config: dict[str, Any], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8") as handle:
        yaml.safe_dump(public_config(config), handle, sort_keys=False, allow_unicode=True)

