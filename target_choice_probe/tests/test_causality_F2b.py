"""F2b: truncation (no look-ahead) for the new pieces: mark 4h bars, asset betas, beta-neutral weights, no-trade band,
timing hint; plus the arithmetic of the beta-neutral scaling and the permutation placebo."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data_F2b import to_4h  # noqa: E402
from src.q2_F2b import permute_across_assets  # noqa: E402
from src.strategies_F2b import asset_betas, band_path, beta_neutral, hint_exec  # noqa: E402

rng = np.random.default_rng(5)
FB = {"q3": {"beta_window_bars": 30, "beta_min_bars": 15}}


def test_mark_4h_truncation_and_rule():
    hours = pd.date_range("2024-01-01", periods=48, freq="h", tz="UTC")
    h = pd.DataFrame({"ts": np.tile(hours, 2), "symbol": np.repeat(["A", "B"], 48),
                      "open": rng.uniform(1, 2, 96), "high": 3.0, "low": 0.5, "close": rng.uniform(1, 2, 96)})
    h = h.drop(index=[60])                                          # B misses one hour -> that 4h bar is NaN
    bars = pd.date_range(hours[0], periods=12, freq="4h")
    full = to_4h(h, bars, ["A", "B"])
    cut = to_4h(h[h.ts < bars[7]], bars[:7], ["A", "B"])
    for k in ("mark_open", "mark_close"):
        assert np.allclose(full[k].iloc[:7], cut[k], equal_nan=True)
    assert full["mark_open"].iloc[0, 0] == h[(h.symbol == "A")].open.iloc[0]
    assert np.isnan(full["mark_close"]["B"].iloc[3]) and not np.isnan(full["mark_close"]["A"].iloc[3])


def test_betas_and_neutral_weights_do_not_look_ahead():
    nb, ns = 200, 6
    R = rng.normal(0, 0.01, (nb, ns))
    inp = {"R_M": R.copy(), "mkt_M": R.mean(1)}
    b1 = asset_betas(inp, FB)
    k = 120
    R2 = R.copy()
    R2[k:] = rng.normal(0, 0.05, (nb - k, ns))
    b2 = asset_betas({"R_M": R2, "mkt_M": R2.mean(1)}, FB)
    assert np.allclose(b1[: k + 1], b2[: k + 1], equal_nan=True)   # beta at t uses rows <= t-1
    dec = np.tile(np.array([0.25, 0.25, 0, 0, -0.25, -0.25]), (nb, 1))
    n1, _ = beta_neutral(dec, b1, 3)
    n2, _ = beta_neutral(dec, b2, 3)
    assert np.allclose(n1[: k + 1], n2[: k + 1])


def test_beta_neutral_arithmetic():
    dec = np.array([[0.25, 0.25, -0.25, -0.25]])
    beta = np.array([[1.5, 1.3, 0.8, 0.6]])
    w, fail = beta_neutral(dec, beta, 1)
    assert fail == 0 and np.isclose(np.abs(w).sum(), 1) and np.isclose((w * beta).sum(), 0)


def test_band_path():
    tgt = np.array([[1.0, 0.0], [1.0, 0.1], [0.0, 0.1]])
    p = band_path(tgt, 0.2, 0.5)
    assert np.allclose(p, [[0.5, 0.0], [0.75, 0.0], [0.375, 0.0]])
    assert np.allclose(band_path(tgt, 0.0, 1.0), tgt)


def test_hint_uses_only_known_forecasts():
    nb, ns = 50, 3
    mo = np.cumprod(1 + rng.normal(0, 0.01, (nb, ns)), axis=0)
    z = rng.normal(size=(nb, ns))
    tr = np.zeros((nb, ns), bool)
    tr[10, 1] = True
    d = np.ones((nb, ns))                                           # a buy
    z[10, 1], z[11, 1] = -3.0, 3.0                                  # postpone once, then execute
    s, dl, _, _ = hint_exec(z, tr, d, mo, 2, 0.5)
    assert dl[0] == 1 and np.isclose(s[0], (mo[11, 1] - mo[12, 1]) / mo[11, 1] * 1e4)
    z2 = z.copy()
    z2[12:] = 99.0                                                  # later forecasts must not matter
    assert np.allclose(hint_exec(z2, tr, d, mo, 2, 0.5)[0], s)


def test_permutation_keeps_cross_section():
    a = rng.normal(size=(20, 7))
    a[3, :2] = np.nan
    o = permute_across_assets(a, rng.random(a.shape))
    assert (np.isnan(o) == np.isnan(a)).all()
    assert np.allclose(np.nanmean(o, 1), np.nanmean(a, 1)) and not np.allclose(o, a, equal_nan=True)
