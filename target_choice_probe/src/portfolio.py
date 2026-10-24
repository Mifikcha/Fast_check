"""Out-of-sample scores -> positions -> daily PnL components (price, funding, costs S0/S1/S2, turnover).

Decision at the close of bar t, rebalanced every h bars (global bar index % h == 0); the weights decided at t are
held over bars t+1 ... t+h (open-to-open). Gross exposure is 1 (sum |w| = 1) whenever the model trades.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from .common import LOGS, WORK, holdout_open, log_attempt, now_utc
from .stats import sharpe
from .walkforward import context, families, load_refits

KEY = lambda br, tgt, h: f"{br}|{tgt}|{h}"


def assemble(refits: list[dict], fam: tuple, shape: tuple, field: str = "pred") -> np.ndarray:
    s = np.full(shape, np.nan, dtype=np.float32)
    for r in refits:
        p = r["fits"][fam][field]
        if p is not None:
            s[r["block"]["T0"]:r["block"]["T1"]] = p
    return s


def decisions(score: np.ndarray, branch: str, h: int, q: float = 0.2, theta: float = 0.0) -> np.ndarray:
    """Target weights at every bar (rows that are not rebalance bars repeat the last decision)."""
    nb, ns = score.shape
    dec = np.zeros((nb, ns))
    last = np.zeros(ns)
    for t in range(nb):
        if t % h == 0:
            s = score[t]
            ok = ~np.isnan(s)
            last = np.zeros(ns)
            if branch == "X" and ok.sum() >= 5:
                k = max(1, int(np.floor(q * ok.sum())))
                order = np.argsort(np.where(ok, s, np.nan))[: ok.sum()]          # ascending, NaN last
                last[order[:k]], last[order[-k:]] = -1.0, 1.0
            elif branch == "D" and ok.any():
                last[ok] = np.where(s[ok] > theta, 1.0, np.where(s[ok] < -theta, -1.0, 0.0))
            g = np.abs(last).sum()
            last = last / g if g > 0 else last
        dec[t] = last
    return dec


def simulate(dec: np.ndarray, R: np.ndarray, fs: np.ndarray, cfg: dict[str, Any], mult: np.ndarray | None = None) -> dict:
    """Bar-level components; held[b] = decision made at b-1. `mult` rescales positions (volatility regulator)."""
    w = dec * mult if mult is not None else dec
    held = np.vstack([np.zeros((1, w.shape[1])), w[:-1]])
    prev = np.vstack([np.zeros((1, w.shape[1])), held[:-1]])
    same = np.sign(held) == np.sign(prev)
    opening = np.where(same, np.maximum(np.abs(held) - np.abs(prev), 0), np.abs(held))
    closing = np.where(same, np.maximum(np.abs(prev) - np.abs(held), 0), np.abs(prev))
    out = {"price": np.nansum(held * np.nan_to_num(R), axis=1), "funding": -np.sum(held * fs, axis=1),
           "turnover": (opening + closing).sum(axis=1), "gross_exposure": np.abs(held).sum(axis=1)}
    for name, c in cfg["costs_bp_per_side"].items():
        out[f"cost_{name}"] = ((opening.sum(1) * (c["entry"] + c["slip"]) + closing.sum(1) * (c["exit"] + c["slip"])) / 1e4)
    return out


def daily(comp: dict, bars: pd.DatetimeIndex, lo: int, hi: int) -> pd.DataFrame:
    df = pd.DataFrame({k: v[lo:hi] for k, v in comp.items()}, index=bars[lo:hi])
    agg = df.drop(columns="gross_exposure").groupby(df.index.floor("D")).sum()
    agg["gross_exposure"] = df["gross_exposure"].groupby(df.index.floor("D")).mean()
    return agg


def market_inputs(ctx: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from .clean import load_panels
    from .features import cut_panels
    pan ={k: v.reindex(ctx["bars"]) for k, v in cut_panels(load_panels(["open", "funding_settled"]), ctx["until_ts"]).items()}
    o = pan["open"].to_numpy()
    R = o[1:] / o[:-1] - 1
    R = np.vstack([R, np.full((1, o.shape[1]), np.nan)])
    fs = np.nan_to_num(pan["funding_settled"].to_numpy())
    mkt = np.nanmean(np.where(ctx["alive"], R, np.nan), axis=1)
    return R, fs, np.nan_to_num(mkt)


def windows(ctx: dict, refits: list[dict]) -> dict[str, tuple[int, int]]:
    """Holding-bar ranges evaluated per period."""
    out = {}
    dev = [r for r in refits if r["block"]["period"] == "dev"]
    out["dev"] = (dev[0]["block"]["T0"] + 1, ctx["dev_end"])
    if holdout_open():
        out["holdout"] = (ctx["hold"] + 1, ctx["until"])
    return out


def run(cfg: dict[str, Any]) -> None:
    from .common import last_usable
    ctx = context(cfg)
    ctx["until_ts"] = last_usable(cfg)
    refits = load_refits()
    shape = ctx["alive"].shape
    R, fs, mkt = market_inputs(ctx)
    win = windows(ctx, refits)
    theta_path = LOGS / "theta_selected_F2.json"
    theta = json.loads(theta_path.read_text(encoding="utf-8")) if theta_path.exists() else None
    frames, sel_rows = [], []
    for br, tgt, h in families(cfg):
        if br == "V":
            continue
        score = assemble(refits, (br, tgt, h), shape)
        grid = cfg["portfolio"]["D"]["theta_grid"] if br == "D" else [None]
        for th in grid:
            dec = decisions(score, br, h, cfg["portfolio"]["X"]["quantile"], th or 0.0)
            comp = simulate(dec, R, fs, cfg)
            for period, (lo, hi) in win.items():
                d = daily(comp, ctx["bars"], lo, hi)
                d["market"] = pd.Series(mkt[lo:hi], index=ctx["bars"][lo:hi]).groupby(ctx["bars"][lo:hi].floor("D")).sum()
                d = d.assign(period=period, branch=br, target=tgt, h=h, theta=th if th is not None else np.nan)
                frames.append(d.rename_axis("date").reset_index())
                if br == "D" and period == "dev":
                    net = d["price"] + d["funding"] - d["cost_S1"]
                    sel_rows.append({"target": tgt, "h": h, "theta": th, "dev_sharpe_S1": sharpe(net)})
    out = pd.concat(frames, ignore_index=True)
    if theta is None:          # chosen on dev only, frozen before the holdout is opened
        s = pd.DataFrame(sel_rows)
        best = s.loc[s.groupby(["target", "h"])["dev_sharpe_S1"].idxmax()]
        theta = {KEY("D", r.target, int(r.h)): float(r.theta) for r in best.itertuples()}
        theta_path.write_text(json.dumps({"selected_utc": now_utc(), "rule": cfg["portfolio"]["D"]["theta_select"],
                                          "theta": theta, "grid_results": sel_rows}, indent=1), encoding="utf-8")
        for r in best.itertuples():
            log_attempt(r.target, int(r.h), "theta", "D:theta_grid", len(cfg["portfolio"]["D"]["theta_grid"]), "dev",
                        f"selected theta={r.theta}")
    else:
        theta = theta["theta"]
    out.to_parquet(WORK / "daily.parquet", index=False)
    print(f"{now_utc()} daily PnL for {out.groupby(['branch','target','h','theta'], dropna=False).ngroups} strategies, periods {list(win)}")
