"""Walk-forward: on every monthly refit score every feature (real + fresh placebo), BH inside each (target, h)
family, keep top-K by |t|, fit the model, predict the next month. Results are cached per block in WORK/refits.

Branch X families score features by the per-bar cross-sectional Spearman IC (mean, Newey-West t, lag h).
Branch D families and T7 score features by one pooled Spearman over (bar, asset), block-bootstrap sd (blocks of h bars).
"""

from __future__ import annotations

import json
import pickle
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import rankdata
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import log_loss

from .clean import load_panels
from .common import CLEAN, WORK, holdout_open, last_usable, log_attempt, now_utc, periods
from .features import cut_panels, open_features, past_log_rv, placebos, targets
from .stats import bh, block_corr, boot_counts, nw_tstat, p_two_sided, rowwise_spearman

X_CLASS = {"T1": "r", "T3": "r", "T4": "r", "T5": "z", "T6": "z"}   # T1/T3/T4 share cross-sectional ranks, so do T5/T6
D_TARGETS = ["T1", "T2", "T5", "T8", "T7"]                         # pooled-scored targets (T7 = volatility)
REFITS = WORK / "refits"
PLACEBO_MM = WORK / "placebo_tmp.f32"


# --------------------------------------------------------------------------- context

def context(cfg: dict[str, Any]) -> dict[str, Any]:
    feats, meta = open_features()
    bars = pd.DatetimeIndex(pd.to_datetime(meta["bars"]))
    full = load_panels()
    until = last_usable(cfg)
    pan = {k: v.reindex(bars) for k, v in cut_panels(full, until).items()}   # NaN after the cut
    hourly = pd.read_parquet(CLEAN / "hourly_logret.parquet")
    hourly = hourly.loc[: until - pd.Timedelta(hours=1)]
    tg = targets(cfg, pan, hourly)
    har = np.stack([past_log_rv(bars, hourly, w) for w in cfg["volatility"]["har_windows_bars"]], -1)
    alive = pan["close"].notna().to_numpy()
    p = periods(cfg)
    idx = lambda ts: int(bars.searchsorted(ts))
    return {"feats": feats, "meta": meta, "bars": bars, "tg": tg, "har": har, "alive": alive, "p": p,
            "a": idx(p["start"]), "dev_end": idx(p["dev_end"]), "hold": idx(p["holdout_start"]), "end": idx(p["end"]),
            "until": idx(until), "kept": np.array(meta["kept"])}


def blocks(cfg: dict[str, Any], ctx: dict[str, Any]) -> list[dict[str, Any]]:
    """Monthly test blocks [T0, T1) of decision bars: dev first, then (only if open) the holdout."""
    p, bars = ctx["p"], ctx["bars"]
    first = p["start"] + pd.DateOffset(months=cfg["walkforward"]["first_train_months"])
    out = []
    for period, lo, hi in (("dev", first, p["dev_end"]), ("holdout", p["holdout_start"], p["end"])):
        if period == "holdout" and not holdout_open():
            break
        edges = [lo] + list(pd.date_range(lo + pd.offsets.MonthBegin(1), hi, freq="MS", inclusive="left")) + [hi]
        edges = sorted(set(pd.Timestamp(e) for e in edges))
        for s, e in zip(edges[:-1], edges[1:]):
            out.append({"label": f"{period}_{s:%Y-%m-%d}", "period": period,
                        "T0": int(bars.searchsorted(s)), "T1": int(bars.searchsorted(e)),
                        "V0": int(bars.searchsorted(s - pd.DateOffset(months=cfg["walkforward"]["alpha_validation_months"])))})
    return out


def train_rows(ctx: dict[str, Any], T0: int, h: int) -> np.ndarray:
    """Decision bars usable for training before a block at T0: target ends by T0 (purge h), embargo never used."""
    t = np.arange(ctx["a"], T0 - h)                       # t + 1 + h <= T0
    return t[(t < ctx["dev_end"]) | (t >= ctx["hold"])]


# --------------------------------------------------------------------------- scoring (runs in worker processes)

def _score_chunk(src: list[tuple[str, int]], shapes: dict, a: int, T0: int, sc: dict) -> list[dict]:
    mms = {k: np.memmap(path, dtype=np.float32, mode="r", shape=shape) for k, (path, shape) in shapes.items()}
    out = []
    for kind, j in src:
        F = np.asarray(mms[kind][j, a:T0], dtype=float)
        res = {}
        for h, d in sc.items():
            for cls, Y in d["X"].items():
                ic = rowwise_spearman(F, Y)
                mu, t = nw_tstat(ic[None, :], h)
                res[("X", cls, h)] = (mu[0], t[0])
            x = np.nan_to_num(F[d["mask"]], nan=0.0)           # missing feature = cross-sectional mean
            c, sd = block_corr(rankdata(x), d["Yr"], d["blk"], d["nblk"], d["C"])
            for k, tgt in enumerate(D_TARGETS):
                res[("D", tgt, h)] = (c[k], c[k] / sd[k] if sd[k] > 0 else np.nan)
        out.append(res)
    return out


def score_inputs(cfg: dict[str, Any], ctx: dict[str, Any], T0: int, seed: int) -> dict:
    a = ctx["a"]
    sc = {}
    rng = np.random.default_rng(seed)
    for h in cfg["horizons_bars"]:
        rows = train_rows(ctx, T0, h)
        use = np.zeros(T0 - a, dtype=bool)
        use[rows - a] = True
        X = {}
        for tgt, cls in (("T1", "r"), ("T5", "z")):
            Y = ctx["tg"][(tgt, h)][a:T0].copy()
            Y[~use] = np.nan
            X[cls] = Y
        Ys = [ctx["tg"][(t, h)][a:T0] for t in D_TARGETS]
        mask = ctx["alive"][a:T0] & use[:, None] & np.all([~np.isnan(y) for y in Ys], axis=0)
        Yr = np.column_stack([rankdata(y[mask]) for y in Ys])
        tt = np.nonzero(mask)[0]
        blk = tt // h
        nblk = int(blk.max()) + 1
        C = boot_counts(nblk, cfg["walkforward"]["bootstrap_reps_selection"], rng)
        sc[h] = {"X": X, "mask": mask, "Yr": Yr, "blk": blk, "nblk": nblk, "C": C}
    return sc


def score_all(cfg, ctx, T0, sc, n_pl) -> list[dict]:
    shapes = {"real": (str(WORK / "features_cand.f32"), tuple(ctx["meta"]["shape"])),
              "placebo": (str(PLACEBO_MM), (n_pl,) + tuple(ctx["meta"]["shape"][1:]))}
    src = [("real", int(j)) for j in ctx["kept"]] + [("placebo", k) for k in range(n_pl)]
    chunks = [src[i:i + 40] for i in range(0, len(src), 40)]
    res = Parallel(n_jobs=14, backend="loky", max_nbytes="1M")(
        delayed(_score_chunk)(c, shapes, ctx["a"], T0, sc) for c in chunks)
    return [r for part in res for r in part]


# --------------------------------------------------------------------------- selection and models

def families(cfg) -> list[tuple[str, str, int]]:
    """(branch, target, h); T7 is scored and selected like branch D."""
    t = cfg["targets"]
    return ([("X", x, h) for x in t["branch_X"] for h in cfg["horizons_bars"]]
            + [("D", x, h) for x in t["branch_D"] for h in cfg["horizons_bars"]]
            + [("V", "T7", h) for h in cfg["horizons_bars"]])


def select(cfg, scores: list[dict], names: list[str], n_real: int, fam: tuple[str, str, int]) -> dict:
    br, tgt, h = fam
    key = ("X", X_CLASS[tgt], h) if br == "X" else ("D", tgt, h)
    stat = np.array([s[key] for s in scores], dtype=float)
    t = stat[:, 1]
    p = p_two_sided(t)
    passed = bh(p, cfg["walkforward"]["bh_q"])
    naive = np.abs(t) > cfg["walkforward"]["naive_abs_t"]
    order = [i for i in np.argsort(-np.nan_to_num(np.abs(t), nan=-1)) if passed[i]][: cfg["walkforward"]["top_k"]]
    is_pl = np.arange(len(t)) >= n_real
    t1 = np.array([n.startswith("placebo1") for n in names])          # time-shifted real features
    t2 = np.array([n.startswith("placebo2") for n in names])          # AR(1) noise
    by_type = {f"type{k}_{what}": int(m.sum()) for k, mask in ((1, t1), (2, t2))
               for what, m in (("n", mask), ("pass", passed & mask), ("naive", naive & mask),
                               ("selected", np.isin(np.arange(len(t)), order) & mask))}
    return by_type | {"family": fam, "n_real": n_real, "n_placebo": int(is_pl.sum()),
            "real_pass": int((passed & ~is_pl).sum()), "placebo_pass": int((passed & is_pl).sum()),
            "real_naive": int((naive & ~is_pl).sum()), "placebo_naive": int((naive & is_pl).sum()),
            "selected": [int(i) for i in order], "selected_names": [names[i] for i in order],
            "selected_placebo": int(sum(is_pl[i] for i in order)),
            "selected_t": [float(t[i]) for i in order], "selected_stat": [float(stat[i, 0]) for i in order]}


def _design(ctx, sel_ids, n_real, pl, lo, hi) -> np.ndarray:
    """[bar, asset, K] of selected features (cs z-scored; missing -> 0 = cross-sectional mean)."""
    cols = []
    for i in sel_ids:
        src = ctx["feats"][ctx["kept"][i]] if i < n_real else pl[i - n_real]
        cols.append(np.nan_to_num(np.asarray(src[lo:hi], dtype=np.float32), nan=0.0))
    return np.stack(cols, -1) if cols else np.zeros((hi - lo, ctx["alive"].shape[1], 0), np.float32)


def fit_family(cfg, ctx, blk, fam, sel, pl) -> dict:
    br, tgt, h = fam
    a, T0, T1, V0 = ctx["a"], blk["T0"], blk["T1"], blk["V0"]
    n_real = sel["n_real"]
    y_all = ctx["tg"][(tgt, h)]
    out = {"family": fam, "alpha": None, "pred": None, "har_pred": None}
    if br != "V" and not sel["selected"]:
        return out                                                    # no finds: no trading this month
    Xf = _design(ctx, sel["selected"], n_real, pl, a, T1)             # rows a..T1
    if br == "V":
        Xf = np.concatenate([ctx["har"][a:T1].astype(np.float32), Xf], -1)
    rows = train_rows(ctx, T0, h)
    tr = np.zeros(ctx["alive"][a:T1].shape, dtype=bool)
    tr[rows - a] = True
    ok = tr & ctx["alive"][a:T1] & ~np.isnan(y_all[a:T1]) & ~np.isnan(Xf).any(-1)
    te = np.zeros_like(ok)
    te[T0 - a:] = ctx["alive"][T0:T1] & ~np.isnan(Xf[T0 - a:]).any(-1)
    val = ok.copy()
    val[: max(V0 - a, 0)] = False
    inner = ok & ~val
    inner[max(V0 - a - h, 0):] = False                                 # purge between inner train and validation
    X_tr, y_tr = Xf[ok], y_all[a:T1][ok]
    kind = "logistic" if tgt in ("T2", "T8") else "ridge"

    def make(alpha):
        return LogisticRegression(C=1.0 / alpha, max_iter=300) if kind == "logistic" else Ridge(alpha=alpha)

    def loss(m, X, y):
        return log_loss(y, m.predict_proba(X), labels=m.classes_) if kind == "logistic" else float(np.mean((m.predict(X) - y) ** 2))

    yv = y_all[a:T1]
    best = None
    for alpha in cfg["walkforward"]["alpha_grid"]:
        m = make(alpha).fit(Xf[inner], yv[inner])
        l = loss(m, Xf[val], yv[val])
        if best is None or l < best[0]:
            best = (l, alpha)
    m = make(best[1]).fit(X_tr, y_tr)
    out["alpha"] = best[1]

    def score(X):
        if kind == "ridge":
            return m.predict(X)
        P = m.predict_proba(X)
        cl = list(m.classes_)
        if tgt == "T2":
            return 2 * P[:, cl.index(1.0)] - 1
        return P[:, cl.index(1.0)] - P[:, cl.index(-1.0)]

    pred = np.full(te[T0 - a:].shape, np.nan, dtype=np.float32)
    pred[te[T0 - a:]] = score(Xf[te])
    if br == "D":                                                      # z-score by the training window
        s_tr = score(X_tr)
        pred = (pred - s_tr.mean()) / s_tr.std()
    out["pred"] = pred
    if br == "V":
        hm = LinearRegression().fit(X_tr[:, :3], y_tr)
        hp = np.full_like(pred, np.nan)
        hp[te[T0 - a:]] = hm.predict(Xf[te][:, :3])
        out["har_pred"] = hp
    return out


# --------------------------------------------------------------------------- driver

def run_block(cfg, ctx, blk, k) -> dict:
    n_real = len(ctx["kept"])
    n_pl = int(round(n_real * cfg["placebo"]["share_of_real"]))
    seed = int(cfg["seed"]) + k
    pl, pl_names = placebos(n_pl, ctx["kept"], ctx["feats"], ctx["alive"], blk["T1"], cfg, seed)
    mm = np.memmap(PLACEBO_MM, dtype=np.float32, mode="w+", shape=pl.shape)
    mm[:] = pl
    mm.flush()
    del mm
    sc = score_inputs(cfg, ctx, blk["T0"], seed)
    scores = score_all(cfg, ctx, blk["T0"], sc, n_pl)
    names = [ctx["meta"]["names"][j] for j in ctx["kept"]] + pl_names
    sels = {fam: select(cfg, scores, names, n_real, fam) for fam in families(cfg)}
    # sequential: thread-parallel fits crashed twice with exit 139 (native segfault) on this machine
    fits = [fit_family(cfg, ctx, blk, f, sels[f], pl) for f in families(cfg)]
    return {"block": blk, "selection": sels, "fits": {f["family"]: f for f in fits}, "n_real": n_real, "n_placebo": n_pl,
            "utc": now_utc()}


def run(cfg: dict[str, Any]) -> None:
    REFITS.mkdir(parents=True, exist_ok=True)
    ctx = context(cfg)
    bl = blocks(cfg, ctx)
    print(f"{now_utc()} {len(bl)} blocks, {len(ctx['kept'])} real features, until bar {ctx['until']}")
    for k, blk in enumerate(bl):
        path = REFITS / f"{blk['label']}.pkl"
        if path.exists():
            continue
        assert blk["T1"] <= ctx["until"], "block reaches data that is not open"
        t0 = pd.Timestamp.now()
        res = run_block(cfg, ctx, blk, k)
        path.write_bytes(pickle.dumps(res))
        s = res["selection"]
        print(f"{now_utc()} {blk['label']} done in {(pd.Timestamp.now() - t0).total_seconds():.0f}s; "
              f"pass X/T1/h6={s[('X','T1',6)]['real_pass']} D/T1/h6={s[('D','T1',6)]['real_pass']} "
              f"placebo X/T1/h6={s[('X','T1',6)]['placebo_pass']}", flush=True)
    period = "holdout" if holdout_open() else "dev"
    n_tests = len(ctx["kept"]) + int(round(len(ctx["kept"]) * cfg["placebo"]["share_of_real"]))
    n_ref = sum(1 for b in bl if b["period"] == period)
    for br, tgt, h in families(cfg):
        model = "logistic" if tgt in ("T2", "T8") else "ridge"
        fam = "grammar+a101" + ("+har" if br == "V" else "")
        log_attempt(tgt, h, fam, f"{br}:{model}", n_tests, period, f"per refit; {n_ref} refits; BH q={cfg['walkforward']['bh_q']}, K={cfg['walkforward']['top_k']}")
        if br == "V":
            log_attempt(tgt, h, "har", "V:har_ols", 3, period, f"baseline, {n_ref} refits")


def load_refits(period: str | None = None) -> list[dict]:
    out = [pickle.loads(p.read_bytes()) for p in sorted(REFITS.glob("*.pkl"))]
    out = [r for r in out if period is None or r["block"]["period"] == period]
    return sorted(out, key=lambda r: r["block"]["T0"])
