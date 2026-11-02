"""Lighter spread and depth from Tardis free samples (first day of each month), streamed, only hourly aggregates kept.

Run from target_choice_probe/: python scripts/lighter_tardis_liquidity.py
quotes          -> time-weighted spread per hour (every BBO change)
book_snapshot_25 -> state on a 10 s grid: spread, depth within +-k bp (thinner side), taker impact vs mid (worse side)
Resumable: one CSV per (dataset, day, market) in data/work/lighter_tardis/.
"""

from __future__ import annotations

import gzip
import io
import json
import sys
import time
import urllib.request
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "work" / "lighter_tardis"
DAYS = ["2026-05-01", "2026-06-01", "2026-07-01", "2026-08-01", "2026-09-01", "2026-10-01", "2026-11-01"]
GRID_US = 10_000_000
BANDS = (5, 10, 25)
SIZES = (10_000, 100_000, 1_000_000)
URL = "https://datasets.tardis.dev/v1/lighter/{t}/{d}/{m}.csv.gz"


def stream(t, day, mid):
    for i in range(5):
        try:
            r = urllib.request.urlopen(URL.format(t=t, d=day.replace("-", "/"), m=mid), timeout=120)
            return io.TextIOWrapper(gzip.GzipFile(fileobj=r), encoding="ascii")
        except Exception:
            time.sleep(5 * 2 ** i)
    raise RuntimeError(f"download failed {t} {day} {mid}")


def quotes(day, mid):
    f = stream("quotes", day, mid)
    next(f)
    acc = {}                                         # hour -> [sum spread*dt, sum dt]
    prev_ts = prev_s = None
    for line in f:
        p = line.split(",")
        ts = int(p[2])
        if prev_ts is not None:
            h = prev_ts // 3_600_000_000
            a = acc.setdefault(h, [0.0, 0.0])
            dt = ts - prev_ts
            a[0] += prev_s * dt
            a[1] += dt
        ask, bid = float(p[5] or "nan"), float(p[6] or "nan")
        s = (ask - bid) / (ask + bid) * 2e4
        prev_ts, prev_s = ts, s if s == s else (prev_s or 0.0)
    return pd.DataFrame([{"hour": h, "spread_tw_bp": v[0] / v[1]} for h, v in acc.items() if v[1] > 0])


def book_metrics(p):
    x = np.array([float(v) if v else np.nan for v in p[4:]]).reshape(-1, 4)  # ask_p, ask_q, bid_p, bid_q
    asks, bids = x[:, :2][~np.isnan(x[:, 0])], x[:, 2:][~np.isnan(x[:, 2])]
    if not len(asks) or not len(bids):
        return None
    mid = (asks[0, 0] + bids[0, 0]) / 2
    r = {"spread_bp": (asks[0, 0] - bids[0, 0]) / mid * 1e4,
         "reach_bp": min(asks[-1, 0] / mid - 1, 1 - bids[-1, 0] / mid) * 1e4}
    for k in BANDS:
        r[f"depth{k}"] = min((asks[:, 0] * asks[:, 1])[asks[:, 0] <= mid * (1 + k / 1e4)].sum(),
                             (bids[:, 0] * bids[:, 1])[bids[:, 0] >= mid * (1 - k / 1e4)].sum())
    for s in SIZES:
        worst = 0.0
        for side in (asks, bids):
            usd = side[:, 0] * side[:, 1]
            cum = np.cumsum(usd)
            if cum[-1] < s:
                worst = np.nan
                break
            j = int(np.searchsorted(cum, s))
            take = np.append(usd[:j], s - (cum[j - 1] if j else 0.0))
            base = (take / side[: j + 1, 0]).sum()
            worst = max(worst, abs(s / base / mid - 1) * 1e4)
        r[f"imp{s // 1000}k"] = worst
    return r


def book(day, mid):
    f = stream("book_snapshot_25", day, mid)
    next(f)
    rows, nxt, last = [], None, None
    for line in f:
        ts = int(line.split(",", 3)[2])
        if nxt is None:
            nxt = (ts // GRID_US + 1) * GRID_US
        while ts >= nxt:                              # state at grid point = last snapshot before it
            if last is not None:
                m = book_metrics(last.rstrip("\n").split(","))
                if m:
                    rows.append(dict(m, t=nxt))
            nxt += GRID_US
        last = line
    g = pd.DataFrame(rows)
    if g.empty:
        return g
    g["hour"] = g.t // 3_600_000_000
    agg = {c: "median" for c in g if c not in ("t", "hour")}
    out = g.groupby("hour").agg(agg)
    for s in SIZES:
        out[f"imp{s // 1000}k_missing"] = g[f"imp{s // 1000}k"].isna().groupby(g.hour).mean()
    out["n_grid"] = g.groupby("hour").size()
    return out.reset_index()


def task(args):
    t, day, sym, mid = args
    path = OUT / f"{t}_{day}_{mid}.csv"
    if path.exists():
        return path.name, "skip"
    t0 = time.time()
    try:
        df = (quotes if t == "quotes" else book)(day, mid)
    except Exception as e:                            # rerun picks it up
        return path.name, f"ERROR {e!r}"
    df.assign(symbol=sym, market_id=mid, day=day).to_csv(path.with_suffix(".tmp"), index=False)
    path.with_suffix(".tmp").replace(path)
    return path.name, f"{time.time() - t0:.0f}s"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    det = json.loads((ROOT / "data" / "raw" / "lighter_orderBookDetails_20261102.json").read_bytes())["order_book_details"]
    ids = {d["symbol"]: d["market_id"] for d in det if d["market_type"] == "perp"}
    ov = pd.read_csv(ROOT / "reports" / "universe_lighter_overlap.csv").dropna(subset=["lighter"])
    jobs = [(t, d, r.bybit, ids[r.lighter]) for t in ("quotes", "book_snapshot_25") for d in DAYS for r in ov.itertuples()]
    with ProcessPoolExecutor(int(sys.argv[1]) if len(sys.argv) > 1 else 10) as ex:
        for i, (name, msg) in enumerate(ex.map(task, jobs), 1):
            print(f"{i}/{len(jobs)} {name} {msg}", flush=True)


if __name__ == "__main__":
    main()
