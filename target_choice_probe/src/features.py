"""Feature library (frozen grammar + optional A101 family), targets T1-T8, placebo features.

Every feature at bar t uses only bars <= t (the bar that closes at the decision time). Values are stored as
float32 in a memmap [feature, bar, asset]; before the holdout is opened, bars after the dev end are left NaN
and are never computed (the panel is cut first, see tests/test_causality.py for the truncation test).
"""

from __future__ import annotations

import json
from typing import Any, Callable

import numpy as np
import pandas as pd
from scipy.signal import lfilter

from .clean import load_panels
from .common import CLEAN, REPORTS, WORK, holdout_open, last_usable, md, now_utc, periods

DF = pd.DataFrame


# --------------------------------------------------------------------------- operators (time series, per asset)

def ts_mean(x: DF, w: int) -> DF: return x.rolling(w, min_periods=w).mean()
def ts_std(x: DF, w: int) -> DF: return x.rolling(w, min_periods=w).std()
def ts_min(x: DF, w: int) -> DF: return x.rolling(w, min_periods=w).min()
def ts_max(x: DF, w: int) -> DF: return x.rolling(w, min_periods=w).max()
def ts_skew(x: DF, w: int) -> DF: return x.rolling(w, min_periods=w).skew()
def ts_rank(x: DF, w: int) -> DF: return x.rolling(w, min_periods=w).rank(pct=True)
def ts_sum(x: DF, w: int) -> DF: return x.rolling(w, min_periods=w).sum()
def delay(x: DF, w: int) -> DF: return x.shift(w)
def delta(x: DF, w: int) -> DF: return x - x.shift(w)
def pct_change(x: DF, w: int) -> DF: return x / x.shift(w) - 1
def ts_corr(x: DF, y: DF, w: int) -> DF: return x.rolling(w, min_periods=w).corr(y).replace([np.inf, -np.inf], np.nan)
def ts_cov(x: DF, y: DF, w: int) -> DF: return x.rolling(w, min_periods=w).cov(y)


def ts_zscore(x: DF, w: int) -> DF:
    return ((x - ts_mean(x, w)) / ts_std(x, w)).replace([np.inf, -np.inf], np.nan)


def decay_linear(x: DF, w: int) -> DF:
    """Weights w, w-1, ..., 1 (newest first) over the last w bars; NaN if any value in the window is NaN."""
    v = x.to_numpy(dtype=float)
    nan = np.isnan(v)
    k = np.arange(w, 0, -1, dtype=float) / (w * (w + 1) / 2)          # k[j] multiplies x[t-j]
    out = np.full_like(v, np.nan)
    z = np.where(nan, 0.0, v)
    cnt = np.vstack([np.zeros((1, v.shape[1])), np.cumsum(nan, axis=0)])
    for j in range(w):
        out[w - 1:] = (out[w - 1:] if j else 0.0) + k[j] * z[w - 1 - j: len(v) - j]
    bad = (cnt[w:] - cnt[:-w]) > 0
    out[w - 1:][bad] = np.nan
    return DF(out, index=x.index, columns=x.columns)


# --------------------------------------------------------------------------- cross-sectional

def cs_zscore(a: np.ndarray, min_n: int = 3) -> np.ndarray:
    with np.errstate(all="ignore"):
        n = np.sum(~np.isnan(a), axis=1, keepdims=True)
        mu = np.nanmean(a, axis=1, keepdims=True)
        sd = np.nanstd(a, axis=1, keepdims=True)
        out = (a - mu) / np.where(sd > 0, sd, np.nan)
    out[(n < min_n).ravel()] = np.nan
    return out


def cs_rank(a: np.ndarray) -> np.ndarray:
    """Percent rank shifted into (-0.5, 0.5): (rank - 0.5) / n - 0.5, ties averaged, NaN kept."""
    r = DF(a).rank(axis=1)
    n = r.notna().sum(axis=1).to_numpy()[:, None]
    return ((r.to_numpy() - 0.5) / n - 0.5)


def rank(x: DF) -> DF:          # A101 `rank`: cross-sectional percent rank
    return x.rank(axis=1, pct=True)


# --------------------------------------------------------------------------- base series

def base_series(pan: dict[str, DF]) -> dict[str, DF]:
    o, h, l, c = pan["open"], pan["high"], pan["low"], pan["close"]
    lg = lambda x: np.log(x.where(x > 0))
    b = {"ret": lg(c / c.shift(1)), "range": h / l - 1, "clv": ((c - l) / (h - l)).where(h > l, 0.5),
         "funding": pan["funding"], "premium": pan["premium"], "oi_chg": lg(pan["oi"] / pan["oi"].shift(1)),
         "log_close": lg(c), "log_open": lg(o), "log_high": lg(h), "log_low": lg(l),
         "log_turnover": lg(pan["turnover"]), "log_volume": lg(pan["volume"]), "log_oi": lg(pan["oi"]),
         "turnover": pan["turnover"], "oi": pan["oi"]}
    b["log_turnover_d1"] = b["log_turnover"].diff()
    b["abs_ret"] = b["ret"].abs()
    return b


# --------------------------------------------------------------------------- grammar

def grammar(cfg: dict[str, Any]) -> list[tuple[str, str, Callable[[dict], DF]]]:
    """Deterministic list of (name, family, fn(base) -> raw DataFrame); cs variants are added in build()."""
    f = cfg["features"]
    W, mw = f["windows"], f["min_window"]
    ops = {"ts_mean": ts_mean, "ts_std": ts_std, "ts_zscore": ts_zscore, "ts_rank": ts_rank, "ts_min": ts_min,
           "ts_max": ts_max, "ts_skew": ts_skew, "decay_linear": decay_linear, "delta": delta}
    out = []
    for s in f["stationary_series"]:
        for op in f["stationary_ops"]:
            out += [(f"{op}({s},{w})", "grammar", lambda b, s=s, op=op, w=w: ops[op](b[s], w))
                    for w in W if w >= mw.get(op, 1)]
    for s in f["level_series"]:
        for op in f["level_ops"]:
            out += [(f"{op}({s},{w})", "grammar", lambda b, s=s, op=op, w=w: ops[op](b[s], w))
                    for w in W if w >= mw.get(op, 1)]
    for s in f["delay_series"]:
        out += [(f"delay({s},{w})", "grammar", lambda b, s=s, w=w: delay(b[s], w)) for w in W]
    for s in f["pct_change_series"]:
        out += [(f"pct_change({s},{w})", "grammar", lambda b, s=s, w=w: pct_change(b[s], w)) for w in W]
    for x, y in f["ts_corr_pairs"]:
        out += [(f"ts_corr({x},{y},{w})", "grammar", lambda b, x=x, y=y, w=w: ts_corr(b[x], b[y], w))
                for w in W if w >= mw["ts_corr"]]
    return out


def a101(pan: dict[str, DF]) -> list[tuple[str, str, Callable[[dict], DF]]]:
    """Kakushadze (2015) formulas without industry neutralisation, translated to crypto (see doc_check_F2.md #8).

    V = turnover (USDT) instead of share volume, so cross-sectional ranks compare like with like;
    price differences that are not scale-free are turned into ratios; returns = bar log return.
    """
    o, h, l, c, V = pan["open"], pan["high"], pan["low"], pan["close"], pan["turnover"]
    vwap = (pan["turnover"] / pan["volume"]).where(pan["volume"] > 0)
    r = np.log(c / c.shift(1))
    oc = (c - o) / o
    safe = lambda d: d.where(d != 0, np.nan)
    F = {
        "A002": lambda: -ts_corr(rank(delta(np.log(V.where(V > 0)), 2)), rank(oc), 6),
        "A006": lambda: -ts_corr(o, V, 10),
        "A009": lambda: (-r).where(~((ts_min(r, 5) > 0) | (ts_max(r, 5) < 0)), r),
        "A012": lambda: np.sign(delta(V, 1)) * (-r),
        "A014": lambda: -rank(delta(r, 3)) * ts_corr(o, V, 10),
        "A018": lambda: -rank(ts_std(oc.abs(), 5) + oc + ts_corr(c, o, 10)),
        "A020": lambda: -(rank(o / h.shift(1) - 1) * rank(o / c.shift(1) - 1) * rank(o / l.shift(1) - 1)),
        "A022": lambda: -delta(ts_corr(h, V, 5), 5) * rank(ts_std(c, 20) / c),
        "A026": lambda: -ts_max(ts_corr(ts_rank(V, 5), ts_rank(h, 5), 5), 3),
        "A033": lambda: rank(-(1 - o / c)),
        "A034": lambda: rank((1 - rank(ts_std(r, 2) / safe(ts_std(r, 5)))) + (1 - rank(r))),
        "A038": lambda: -rank(ts_rank(c, 10)) * rank(c / o),
        "A040": lambda: -rank(ts_std(h, 10) / h) * ts_corr(h, V, 10),
        "A041": lambda: np.sqrt(h * l) / vwap - 1,
        "A044": lambda: -ts_corr(h, rank(V), 5),
        "A053": lambda: -delta(((c - l) - (h - c)) / safe(c - l), 9),
        "A054": lambda: -((l - c) * o ** 5) / safe((l - h) * c ** 5),
        "A055": lambda: -ts_corr(rank((c - ts_min(l, 12)) / safe(ts_max(h, 12) - ts_min(l, 12))), rank(V), 6),
        "A101": lambda: oc / ((h - l) / o + 0.001),
    }
    return [(k, "a101", lambda b, fn=fn: fn()) for k, fn in F.items()]


def candidates(cfg: dict[str, Any], pan: dict[str, DF]) -> list[tuple[str, str, str, Callable]]:
    """(name, family, cs_variant, fn) in the frozen order used by the greedy dedupe."""
    out = [(f"{v}:{n}", fam, v, fn) for n, fam, fn in grammar(cfg) for v in cfg["features"]["cs_variants"]]
    if cfg["features"]["a101"]:
        out += [(f"cs_zscore:{n}", fam, "cs_zscore", fn) for n, fam, fn in a101(pan)]
    if len(out) > cfg["features"]["max_features"]:
        raise ValueError(f"{len(out)} candidates > max_features")
    return out


def apply_cs(raw: DF, variant: str) -> np.ndarray:
    a = raw.to_numpy(dtype=float)
    a[~np.isfinite(a)] = np.nan
    return cs_zscore(cs_rank(a) if variant == "cs_rank" else a).astype(np.float32)


def cut_panels(pan: dict[str, DF], until: pd.Timestamp) -> dict[str, DF]:
    """Panels with every bar that closes after `until` dropped (a bar [t, t+4h) closes at t+4h)."""
    return {k: v.loc[: until - pd.Timedelta(hours=4)] for k, v in pan.items()}


# --------------------------------------------------------------------------- targets

def targets(cfg: dict[str, Any], pan: dict[str, DF], hourly: DF | None = None) -> dict[tuple[str, int], np.ndarray]:
    """Decision at the close of bar t; entry at open(t+1), exit at open(t+1+h). Arrays [bar, asset]."""
    t = cfg["targets"]
    o = pan["open"]
    ret = np.log(pan["close"] / pan["close"].shift(1))
    sig = ret.rolling(t["sigma_window_bars"], min_periods=t["sigma_window_bars"]).std().to_numpy()
    out = {}
    for h in cfg["horizons_bars"]:
        r = np.log(o.shift(-(1 + h)) / o.shift(-1)).to_numpy()
        z = r / (sig * np.sqrt(h))
        z[~np.isfinite(z)] = np.nan
        with np.errstate(all="ignore"):
            out[("T1", h)] = r
            out[("T2", h)] = np.sign(r)
            out[("T3", h)] = r - np.nanmean(r, axis=1, keepdims=True)
            out[("T4", h)] = cs_rank(r)
            out[("T5", h)] = z
            out[("T6", h)] = cs_rank(z)
            c = t["round_trip_cost"]
            out[("T8", h)] = np.where(np.isnan(r), np.nan, np.where(r > c, 1.0, np.where(r < -c, -1.0, 0.0)))
        if hourly is not None:
            out[("T7", h)] = future_log_rv(pan["open"].index, hourly, h)
    return out


def _hour_cumsum(bars: pd.DatetimeIndex, hourly: DF) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    hr = hourly.reindex(columns=hourly.columns)
    sq = hr.to_numpy() ** 2
    nan = np.isnan(sq)
    cs = np.vstack([np.zeros((1, sq.shape[1])), np.cumsum(np.where(nan, 0, sq), axis=0)])
    cn = np.vstack([np.zeros((1, sq.shape[1])), np.cumsum(nan, axis=0)])
    return cs, cn, hr.index


def future_log_rv(bars: pd.DatetimeIndex, hourly: DF, h: int) -> np.ndarray:
    """ln sum of squared hourly log returns over [open(t+1), open(t+1+h)) = 4h hours; NaN if any hour is missing."""
    cs, cn, idx = _hour_cumsum(bars, hourly)
    lo = idx.searchsorted(bars + pd.Timedelta(hours=4))
    hi = lo + 4 * h
    ok = hi <= len(idx)
    out = np.full((len(bars), cs.shape[1]), np.nan)
    rv = cs[hi[ok]] - cs[lo[ok]]
    bad = (cn[hi[ok]] - cn[lo[ok]]) > 0
    with np.errstate(all="ignore"):
        out[ok] = np.where(bad | (rv <= 0), np.nan, np.log(rv))
    return out


def past_log_rv(bars: pd.DatetimeIndex, hourly: DF, w_bars: int) -> np.ndarray:
    """HAR regressor: ln mean squared hourly return over the 4*w hours that end at the close of bar t."""
    cs, cn, idx = _hour_cumsum(bars, hourly)
    hi = idx.searchsorted(bars + pd.Timedelta(hours=4))
    lo = hi - 4 * w_bars
    ok = lo >= 0
    out = np.full((len(bars), cs.shape[1]), np.nan)
    rv = (cs[hi[ok]] - cs[lo[ok]]) / (4 * w_bars)
    bad = (cn[hi[ok]] - cn[lo[ok]]) > 4 * w_bars * 0.1       # allow <= 10% missing hours in the past window
    with np.errstate(all="ignore"):
        out[ok] = np.where(bad | (rv <= 0), np.nan, np.log(rv))
    return out


# --------------------------------------------------------------------------- placebo

def placebos(n: int, real_idx: np.ndarray, feats: np.memmap, alive: np.ndarray, n_avail: int,
             cfg: dict[str, Any], seed: int) -> tuple[np.ndarray, list[str]]:
    """n placebo features [n, bar, asset]: half time-shifted real features, half AR(1) noise.

    Only bars < n_avail are used or filled (the holdout stays untouched while closed).
    """
    rng = np.random.default_rng(seed)
    nb, ns = alive.shape
    out = np.full((n, nb, ns), np.nan, dtype=np.float32)
    names = []
    min_shift = cfg["placebo"]["type1_min_shift_days"] * 24 // cfg["bar_hours"]
    n1 = n // 2
    for k in range(n1):
        j = int(rng.choice(real_idx))
        src = np.asarray(feats[j, :n_avail])
        for a in range(ns):
            s = int(rng.integers(min_shift, max(min_shift + 1, n_avail - min_shift)))
            out[k, :n_avail, a] = np.roll(src[:, a], s)
        names.append(f"placebo1:shift:{j}")
    phis = cfg["placebo"]["type2_phi"]
    for k in range(n1, n):
        phi = phis[(k - n1) % len(phis)]
        e = rng.standard_normal((n_avail, ns))
        e[0] /= np.sqrt(1 - phi ** 2)                       # start in the stationary distribution
        out[k, :n_avail] = lfilter([1.0], [1.0, -phi], e, axis=0)
        names.append(f"placebo2:ar1:{phi}")
    out[:, ~alive] = np.nan
    for k in range(n):
        out[k, :n_avail] = cs_zscore(out[k, :n_avail].astype(float))
    return out, names


# --------------------------------------------------------------------------- build

def open_features(mode: str = "r") -> tuple[np.memmap, dict[str, Any]]:
    meta = json.loads((WORK / "features_meta.json").read_text(encoding="utf-8"))
    mm = np.memmap(WORK / "features_cand.f32", dtype=np.float32, mode=mode, shape=tuple(meta["shape"]))
    return mm, meta


def build(cfg: dict[str, Any]) -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    full = load_panels()
    until = last_usable(cfg)
    pan = cut_panels(full, until)
    nb_full, ns = full["close"].shape
    nb = len(pan["close"])
    b = base_series(pan)
    cands = candidates(cfg, pan)
    meta_path = WORK / "features_meta.json"
    old = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else None
    shape = (len(cands), nb_full, ns)
    check = None
    if old is not None:     # keep a sample of the old (dev) values to prove the rebuild does not change them
        nb_old = int(full["close"].index.searchsorted(pd.Timestamp(old["computed_until"])))
        ids = old["kept"][::25]
        check = (ids, nb_old, np.array(open_features()[0][ids, :nb_old]))
    mm = np.memmap(WORK / "features_cand.f32", dtype=np.float32, mode="w+", shape=shape)
    t0 = pd.Timestamp.now()
    for i, (name, fam, var, fn) in enumerate(cands):
        mm[i, :nb] = apply_cs(fn(b), var)
        mm[i, nb:] = np.nan
        if i % 100 == 0:
            print(f"{now_utc()} feature {i}/{len(cands)} {name}")
    mm.flush()
    print(f"computed {len(cands)} candidates in {(pd.Timestamp.now() - t0).total_seconds():.0f}s up to {until}")
    if old is not None:     # after the holdout opens: same frozen kept list, dev part must be identical
        assert [c[0] for c in cands] == old["names"], "grammar changed"
        ids, nb_old, before = check
        assert np.allclose(before, mm[ids, :nb_old], equal_nan=True, atol=1e-5), "dev feature values changed"
        print(f"rebuild check: {len(ids)} features x {nb_old} dev bars identical")
        meta = old | {"computed_until": str(until), "rebuilt_utc": now_utc()}
    else:
        meta = {"names": [c[0] for c in cands], "family": [c[1] for c in cands], "shape": list(shape),
                "bars": [str(x) for x in full["close"].index], "symbols": list(full["close"].columns),
                "computed_until": str(until), "built_utc": now_utc()}
        meta |= dedupe(cfg, mm, meta, full["close"].index)
    meta_path.write_text(json.dumps(meta, indent=0), encoding="utf-8")
    if old is None:
        write_library_report(cfg, meta)


def dedupe(cfg: dict[str, Any], mm: np.memmap, meta: dict[str, Any], bars: pd.DatetimeIndex) -> dict[str, Any]:
    """Target-free greedy dedupe (|corr| > 0.98 with a kept feature) and PCA effective number, dev bars only."""
    p = periods(cfg)
    every = cfg["features"]["dedupe_sample_every_bars"]
    rows = np.arange(bars.searchsorted(p["start"]), bars.searchsorted(p["dev_end"] - pd.Timedelta(hours=4)), every)
    n = mm.shape[0]
    X = np.empty((len(rows) * mm.shape[2], n), dtype=np.float32)
    for i in range(n):
        X[:, i] = np.nan_to_num(np.asarray(mm[i, rows]).ravel(), nan=0.0)
    X -= X.mean(axis=0)
    sd = X.std(axis=0)
    const = sd < 1e-8
    X /= np.where(const, 1, sd)
    C = (X.T @ X) / len(X)
    del X
    keep: list[int] = []
    dropped_by = {}
    for i in range(n):
        if const[i]:
            dropped_by[i] = "constant"
            continue
        if keep:
            j = int(np.argmax(np.abs(C[i, keep])))
            if abs(C[i, keep[j]]) > cfg["features"]["dedupe_abs_corr"]:
                dropped_by[i] = keep[j]
                continue
        keep.append(i)
    ev = np.sort(np.linalg.eigvalsh(C[np.ix_(keep, keep)].astype(np.float64)))[::-1]
    n_eff = int(np.searchsorted(np.cumsum(ev) / ev.sum(), cfg["features"]["pca_variance"]) + 1)
    return {"kept": keep, "n_kept": len(keep), "n_candidates": n, "n_dropped": n - len(keep),
            "dropped_by": {str(k): (meta["names"][v] if isinstance(v, int) else v) for k, v in dropped_by.items()},
            "n_eff_pca95": n_eff, "dedupe_rows": int(len(rows))}


def write_library_report(cfg: dict[str, Any], meta: dict[str, Any]) -> None:
    fam = pd.Series(meta["family"])
    kept = pd.Series(meta["family"]).iloc[meta["kept"]]
    tab = pd.DataFrame({"кандидатов": fam.value_counts(), "оставлено": kept.value_counts()}).fillna(0).astype(int)
    var = pd.Series([n.split(":")[0] for n in meta["names"]])
    tab2 = pd.DataFrame({"кандидатов": var.value_counts(), "оставлено": var.iloc[meta["kept"]].value_counts()}).fillna(0).astype(int)
    f = cfg["features"]
    text = f"""# Библиотека признаков (Stage F2)

Записано {meta['built_utc']}, **до** любых расчётов с целями прогноза.

- Кандидатов по замороженной грамматике: **{meta['n_candidates']}** (предел {f['max_features']}).
- Удалено как дубли (|корреляция| > {f['dedupe_abs_corr']} с уже оставленным признаком; без использования цели,
  только бары разработки, каждый {f['dedupe_sample_every_bars']}-й бар, {meta['dedupe_rows']} баров × все активы): **{meta['n_dropped']}**.
- Оставлено признаков: **{meta['n_kept']}**.
- Эффективное число независимых проверок (главных компонент на {int(f['pca_variance']*100)}% дисперсии оставленных признаков): **{meta['n_eff_pca95']}**.

Плацебо на каждом переобучении: {int(cfg['placebo']['share_of_real']*100)}% от {meta['n_kept']} = {int(round(meta['n_kept']*cfg['placebo']['share_of_real']))} шт.

## По семействам

{md(tab)}

## По варианту среза активов

`cs_zscore` — z-оценка по срезу активов; `cs_rank` — сначала ранг по срезу, потом z-оценка. Перед моделью всё равно делается cs_zscore.

{md(tab2)}

## Грамматика (заморожена в `configs/stageF2_frozen.yaml`)

- Базовые ряды (4-часовые бары): доходность бара `ret`, диапазон `range` = high/low − 1, положение закрытия в диапазоне `clv`,
  фандинг в час `funding` (последний расчётный строго внутри бара), премиум-индекс `premium` (среднее часовых закрытий в баре),
  изменение OI `oi_chg`, логарифмы open/high/low/close, оборота, объёма и OI, сырые оборот и OI.
- Стационарные ряды × {{{', '.join(f['stationary_ops'])}}} × окна {f['windows']}.
- Уровни (логарифмы) × {{{', '.join(f['level_ops'])}}} × окна.
- delay(ret, w), pct_change(turnover|oi, w), ts_corr по парам {f['ts_corr_pairs']}.
- Минимальные окна: {f['min_window']}.
- Семейство A101: {sum(1 for x in meta['family'] if x == 'a101')} формул из Kakushadze (2015), перевод на крипту описан в `doc_check_F2.md` (п. 8).

Признак называется `вариант:оператор(ряд,окно)`, например `cs_rank:ts_zscore(funding,42)`.
"""
    (REPORTS / "feature_library.md").write_text(text, encoding="utf-8")
    print(text[:1200])
