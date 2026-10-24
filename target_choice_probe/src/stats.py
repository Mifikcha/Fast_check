"""Small statistics toolbox: BH, Holm, Newey-West t, row-wise Spearman, block sums, Sharpe, drawdown, ADF."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as st


def bh(p: np.ndarray, q: float) -> np.ndarray:
    """Benjamini-Hochberg: boolean mask of discoveries at FDR q."""
    p = np.asarray(p, dtype=float)
    m = np.sum(~np.isnan(p))
    out = np.zeros(len(p), dtype=bool)
    if m == 0:
        return out
    order = np.argsort(np.where(np.isnan(p), np.inf, p))
    ps = p[order][:m]
    ok = np.nonzero(ps <= q * np.arange(1, m + 1) / m)[0]
    if len(ok):
        out[order[: ok[-1] + 1]] = True
    return out


def holm(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj = np.maximum.accumulate((m - np.arange(m)) * p[order])
    out = np.empty(m)
    out[order] = np.minimum(adj, 1.0)
    return out


def nw_tstat(x: np.ndarray, lag: int) -> tuple[np.ndarray, np.ndarray]:
    """Rows = series (NaN = missing). Mean and Newey-West t with Bartlett weights 1 - l/(L+1)."""
    x = np.atleast_2d(np.asarray(x, dtype=float))
    ok = ~np.isnan(x)
    n = ok.sum(axis=1)
    mu = np.nanmean(x, axis=1)
    e = np.where(ok, x - mu[:, None], 0.0)
    var = (e * e).sum(axis=1) / n
    for l in range(1, lag + 1):
        var += 2 * (1 - l / (lag + 1)) * (e[:, l:] * e[:, :-l]).sum(axis=1) / n
    with np.errstate(all="ignore"):
        t = mu / np.sqrt(var / n)
    t[n < 30] = np.nan
    return mu, t


def p_two_sided(t: np.ndarray) -> np.ndarray:
    return 2 * st.norm.sf(np.abs(t))


def rowwise_spearman(f: np.ndarray, y: np.ndarray, min_n: int = 10) -> np.ndarray:
    """Spearman correlation across assets for every bar (row); ranks over pairs where both are present."""
    bad = np.isnan(f) | np.isnan(y)
    fr = pd.DataFrame(np.where(bad, np.nan, f)).rank(axis=1).to_numpy()
    yr = pd.DataFrame(np.where(bad, np.nan, y)).rank(axis=1).to_numpy()
    n = (~bad).sum(axis=1)
    with np.errstate(all="ignore"):
        fr -= np.nanmean(fr, axis=1, keepdims=True)
        yr -= np.nanmean(yr, axis=1, keepdims=True)
        c = np.nansum(fr * yr, axis=1) / np.sqrt(np.nansum(fr * fr, axis=1) * np.nansum(yr * yr, axis=1))
    c[n < min_n] = np.nan
    return c


def block_corr(x: np.ndarray, Y: np.ndarray, blk: np.ndarray, nblk: int, C: np.ndarray | None):
    """Pooled Pearson of x with each column of Y (both already ranks), plus bootstrap sd over blocks.

    C: [reps, nblk] counts of each block in each bootstrap draw (non-overlapping blocks, with replacement).
    The correlation from block sums equals the correlation of the concatenated resample.
    """
    s = lambda w: np.bincount(blk, weights=w, minlength=nblk)
    n, sx, sxx = s(np.ones_like(x)), s(x), s(x * x)
    sy = np.stack([s(Y[:, j]) for j in range(Y.shape[1])], 1)
    syy = np.stack([s(Y[:, j] ** 2) for j in range(Y.shape[1])], 1)
    sxy = np.stack([s(x * Y[:, j]) for j in range(Y.shape[1])], 1)

    def corr(N, Sx, Sxx, Sy, Syy, Sxy):
        with np.errstate(all="ignore"):
            return (N * Sxy - Sx * Sy) / np.sqrt((N * Sxx - Sx ** 2) * (N * Syy - Sy ** 2))

    c = corr(n.sum(), sx.sum(), sxx.sum(), sy.sum(0), syy.sum(0), sxy.sum(0))
    if C is None:
        return c, None
    b = corr((C @ n)[:, None], (C @ sx)[:, None], (C @ sxx)[:, None], C @ sy, C @ syy, C @ sxy)
    return c, np.nanstd(b, axis=0, ddof=1)


def boot_counts(nblk: int, reps: int, rng: np.random.Generator) -> np.ndarray:
    idx = rng.integers(0, nblk, size=(reps, nblk))
    C = np.zeros((reps, nblk), dtype=np.float64)
    np.add.at(C, (np.repeat(np.arange(reps), nblk), idx.ravel()), 1.0)
    return C


def sharpe(d: np.ndarray) -> float:
    d = np.asarray(d, dtype=float)
    sd = d.std(ddof=1)
    return float(d.mean() / sd * np.sqrt(365)) if sd > 0 else float("nan")


def max_drawdown(d: np.ndarray) -> float:
    eq = np.cumsum(np.asarray(d, dtype=float))          # additive daily returns on gross
    return float(np.max(np.maximum.accumulate(np.r_[0, eq]) - np.r_[0, eq]))


def block_boot_days(n: int, block: int, reps: int, rng: np.random.Generator) -> np.ndarray:
    """Index matrix [reps, n] of a non-overlapping-block bootstrap of n days (last block may be short)."""
    starts = np.arange(0, n, block)
    pick = rng.integers(0, len(starts), size=(reps, int(np.ceil(n / block))))
    idx = (starts[pick][:, :, None] + np.arange(block)[None, None, :]).reshape(reps, -1)
    idx = np.where(idx < n, idx, idx - n)    # wrap the short last block (circular)
    return idx[:, :n]


def adf(y: np.ndarray, max_lag: int | None = None) -> dict:
    """ADF with constant, lag by AIC in [0, 12 (n/100)^0.25]; asymptotic MacKinnon critical values."""
    y = np.asarray(y, dtype=float)
    y = y[~np.isnan(y)]
    n = len(y)
    max_lag = int(12 * (n / 100) ** 0.25) if max_lag is None else max_lag
    dy = np.diff(y)
    best = None
    for p in range(max_lag + 1):
        Y = dy[max_lag:]
        X = [np.ones(len(Y)), y[max_lag:-1]] + [dy[max_lag - k: len(dy) - k] for k in range(1, p + 1)]
        X = np.column_stack(X)
        beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
        res = Y - X @ beta
        s2 = res @ res / (len(Y) - X.shape[1])
        aic = len(Y) * np.log(res @ res / len(Y)) + 2 * X.shape[1]
        if best is None or aic < best[0]:
            se = np.sqrt(s2 * np.linalg.inv(X.T @ X)[1, 1])
            best = (aic, beta[1] / se, p)
    return {"stat": float(best[1]), "lag": best[2], "n": n, "crit": {"1%": -3.43, "5%": -2.86, "10%": -2.57}}
