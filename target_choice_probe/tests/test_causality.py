"""Truncation test (no look-ahead) for every feature formula and the targets, plus the statistics helpers."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import load_config  # noqa: E402
from src.features import apply_cs, base_series, candidates, cut_panels, decay_linear, future_log_rv, past_log_rv, targets  # noqa: E402
from src.portfolio import decisions, simulate  # noqa: E402
from src.stats import adf, bh, block_corr, boot_counts, holm, nw_tstat, rowwise_spearman  # noqa: E402

rng = np.random.default_rng(11)
CFG, _ = load_config()


def fake_panels(nb=700, ns=12):
    bars = pd.date_range("2022-01-01", periods=nb, freq="4h", tz="UTC")
    syms = [f"S{i}" for i in range(ns)]
    lc = np.cumsum(rng.normal(0, 0.02, (nb, ns)), axis=0) + np.log(rng.uniform(0.1, 100, ns))
    c = np.exp(lc)
    o = c * np.exp(rng.normal(0, 0.005, (nb, ns)))
    h = np.maximum(o, c) * np.exp(np.abs(rng.normal(0, 0.01, (nb, ns))))
    l = np.minimum(o, c) * np.exp(-np.abs(rng.normal(0, 0.01, (nb, ns))))
    vol = rng.lognormal(10, 1, (nb, ns))
    d = {"open": o, "high": h, "low": l, "close": c, "volume": vol, "turnover": vol * c,
         "premium": rng.normal(0, 1e-4, (nb, ns)), "oi": rng.lognormal(12, 0.3, (nb, ns)),
         "funding": rng.normal(1e-5, 1e-5, (nb, ns)), "funding_settled": np.zeros((nb, ns))}
    pan = {k: pd.DataFrame(v, index=bars, columns=syms) for k, v in d.items()}
    pan["close"].iloc[:150, 3] = np.nan            # a late listing
    for k in ("open", "high", "low", "volume", "turnover"):
        pan[k].iloc[:150, 3] = np.nan
    hours = pd.date_range(bars[0], periods=nb * 4, freq="h", tz="UTC")
    hourly = pd.DataFrame(rng.normal(0, 0.01, (nb * 4, ns)), index=hours, columns=syms)
    return pan, hourly


def test_features_truncation():
    pan, _ = fake_panels()
    full = base_series(pan)
    cands = candidates(CFG, pan)
    for cut in (420, 611):
        until = pan["close"].index[cut] + pd.Timedelta(hours=4)         # bar `cut` is the last one kept
        pc = cut_panels(pan, until)
        assert len(pc["close"]) == cut + 1
        bc = base_series(pc)
        cc = candidates(CFG, pc)
        for (name, _, var, fn), (_, _, _, fn_c) in zip(cands, cc):
            a = apply_cs(fn(full), var)[: cut + 1]
            b = apply_cs(fn_c(bc), var)
            assert np.allclose(a, b, equal_nan=True, atol=1e-5), name


def test_targets_use_next_open_and_truncate():
    pan, hourly = fake_panels()
    tg = targets(CFG, pan, hourly)
    o = pan["open"].to_numpy()
    t, i, h = 300, 2, 6
    assert np.isclose(tg[("T1", h)][t, i], np.log(o[t + 1 + h, i] / o[t + 1, i]))
    # a target at t needs bars up to t+1+h; cutting there leaves it unchanged, cutting earlier makes it NaN
    until = pan["close"].index[t + 1 + h] + pd.Timedelta(hours=4)
    tc = targets(CFG, cut_panels(pan, until), hourly.loc[: until - pd.Timedelta(hours=1)])
    assert np.isclose(tc[("T1", h)][t, i], tg[("T1", h)][t, i])
    assert np.isnan(tc[("T1", h)][t + 1, i])
    assert np.allclose(np.nanmax(np.abs(tg[("T4", h)])), 0.5, atol=0.05)
    # T7 window = hours of bars t+1 .. t+h
    hr = hourly.to_numpy()[4 * (t + 1): 4 * (t + 1 + h), i]
    assert np.isclose(tg[("T7", h)][t, i], np.log(np.sum(hr ** 2)))
    past = past_log_rv(pan["open"].index, hourly, 6)
    assert np.isclose(past[t, i], np.log(np.mean(hourly.to_numpy()[4 * (t + 1) - 24: 4 * (t + 1), i] ** 2)))


def test_decay_linear():
    x = pd.DataFrame({"a": [1.0, 2, 3, 4, np.nan, 6, 7, 8]})
    d = decay_linear(x, 3)["a"].to_numpy()
    assert np.isclose(d[3], (4 * 3 + 3 * 2 + 2 * 1) / 6) and np.isnan(d[4]) and np.isnan(d[6]) and np.isclose(d[7], (8 * 3 + 7 * 2 + 6) / 6)


def test_stats_helpers():
    p = np.array([0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205, 0.212, 0.216])
    assert bh(p, 0.05).sum() == 2                                      # textbook BH example
    assert np.allclose(holm(np.array([0.01, 0.04, 0.03])), [0.03, 0.06, 0.06])
    x = rng.normal(0.1, 1, (1, 4000))
    mu, t = nw_tstat(x, 0)
    assert abs(t[0] - 0.1 * np.sqrt(4000)) < 1.5
    f, y = rng.normal(size=(50, 30)), rng.normal(size=(50, 30))
    ic = rowwise_spearman(f, f * 2 + 1)
    assert np.allclose(ic, 1.0)
    xr, Y = rng.normal(size=2000), rng.normal(size=(2000, 2))
    blk = np.arange(2000) // 5
    c, sd = block_corr(xr, Y, blk, 400, boot_counts(400, 300, rng))
    assert np.allclose(c, [np.corrcoef(xr, Y[:, j])[0, 1] for j in range(2)])
    assert np.all(np.abs(sd - 1 / np.sqrt(2000)) < 0.008)
    rw = np.cumsum(rng.normal(size=1500))
    assert adf(rw)["stat"] > -2.86 and adf(rng.normal(size=1500))["stat"] < -3.43


def test_portfolio_costs_and_timing():
    nb, ns = 12, 10
    score = np.tile(np.arange(ns, dtype=float), (nb, 1))
    dec = decisions(score, "X", 3)
    assert np.isclose(np.abs(dec[0]).sum(), 1) and dec[0, -1] > 0 and dec[0, 0] < 0
    R = np.zeros((nb, ns))
    R[1, -1] = 0.10                                                    # bar 1 is the first held bar
    comp = simulate(dec, R, np.zeros((nb, ns)), CFG)
    assert np.isclose(comp["price"][1], 0.25 * 0.10) and comp["price"][0] == 0
    assert np.isclose(comp["turnover"][1], 1.0) and np.isclose(comp["cost_S1"][1], 6.5e-4)
    assert np.isclose(comp["turnover"][4], 0.0)                       # same weights at the next rebalance: no trade
