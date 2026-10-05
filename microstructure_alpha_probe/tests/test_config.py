from __future__ import annotations

from copy import deepcopy

import pytest

from src.config import load_config, validate_config


def test_default_config_is_valid_and_costs_are_scenarios() -> None:
    config = load_config()
    assert max(config["targets"]["horizons_s"]) <= config["splits"]["embargo_s"]
    assert config["trading"]["incremental_round_trip_cost_bps"] == [0.0, 2.0, 5.0, 10.0, 15.0]


def test_rejects_leaky_short_embargo() -> None:
    config = load_config()
    bad = deepcopy(config)
    bad["splits"]["embargo_s"] = 30
    with pytest.raises(ValueError, match="maximum horizon"):
        validate_config(bad)

