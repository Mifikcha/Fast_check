"""Raw daily files back into arrays for the paper trades."""
import collections
import datetime as dt
import json

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .match import category, parse_base, unit_price
from .paper import Leg, Pair, vwap
from .recorder import RAW


def days_recorded():
    folder = RAW / "snaps"
    return sorted(d.name for d in folder.glob("*") if d.is_dir()) if folder.exists() else []


def segments(cfg, days=None):
    """First full days are the development sample, the rest control. The first calendar day is partial and is left out."""
    today = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")
    days = [d for d in (days_recorded() if days is None else days) if d < today]  # today is still being written
    full = days[1:]
    n = cfg["samples"]["dev_days"]
    return full[:n], full[n:]


def day_bounds(day):
    t0 = int(dt.datetime.fromisoformat(day).replace(tzinfo=dt.timezone.utc).timestamp())
    return t0, t0 + 86400


def read(table, venue, days, flt=None, columns=None):
    out = []
    for d in days:
        folder = RAW / table / d
        for f in list(folder.glob(f"{venue}.parquet")) + list(folder.glob(f"{venue}.*.shard")):
            out.append(pq.read_table(f, filters=flt, columns=columns))
    return pa.concat_tables(out).to_pandas() if out else pd.DataFrame()


def instruments():
    """Latest known instrument of each leg."""
    out = {}
    for f in sorted((RAW / "insts").glob("*.json")):
        for r in json.loads(f.read_text(encoding="utf-8")):
            out[r["venue"], r["kind"], r["native"]] = r
    return out


def episodes():
    """One row per candidate episode, from the recorder's own log."""
    path = RAW / "candidates.csv"
    if not path.exists():
        return pd.DataFrame(columns=["asset", "long", "short", "start", "end", "s_in_bp"])
    open_ep, out = {}, []
    for r in pd.read_csv(path).itertuples():
        key = (r.long, r.short)
        if r.event == "start":
            open_ep[key] = (r.asset, r.t, r.s_in_bp)
        elif key in open_ep:
            a, t, s = open_ep.pop(key)
            out.append((a, *key, t, r.t, s))
    out += [(a, *key, t, np.nan, s) for key, (a, t, s) in open_ep.items()]
    return pd.DataFrame(out, columns=["asset", "long", "short", "start", "end", "s_in_bp"])


def usdc_rate(cfg, days):
    v, kind, native = cfg["recording"]["usdc_leg"]
    q = read("quotes", v, days, [("native", "=", native)], ["t", "bid", "ask"])
    if q.empty:
        return np.array([0.0]), np.array([1.0])
    q = q.sort_values("t")
    return q["t"].to_numpy(float), ((q["bid"] + q["ask"]) / 2).to_numpy(float)


def settled_funding(venue, native, days):
    """Actual settlements where the venue publishes them, else the last snapshot before each settlement (the rate is then an estimate)."""
    h = read("fund_hist", venue, days, [("native", "=", native)], ["ms", "rate"])
    if not h.empty:
        h = h.drop_duplicates("ms").sort_values("ms")
        return (h["ms"] / 1000).to_numpy(float), h["rate"].to_numpy(float), True
    f = read("funding", venue, days, [("native", "=", native)], ["t", "rate", "next_ms", "interval_h"])
    if f.empty:
        return np.empty(0), np.empty(0), False
    f = f.sort_values("t")
    if f["next_ms"].notna().any():
        f = f.dropna(subset=["next_ms"]).groupby("next_ms", as_index=False).last()
        return (f["next_ms"] / 1000).to_numpy(float), f["rate"].to_numpy(float), False
    iv = (f["interval_h"].fillna(8) * 3600).to_numpy()  # Paradex accrues continuously: a settlement per interval on the UTC grid
    f["slot"] = (f["t"].to_numpy() // iv + 1) * iv
    f = f.groupby("slot", as_index=False).last()
    return f["slot"].to_numpy(float), f["rate"].to_numpy(float), False


def build_leg(key, inst, cfg, days, q_usd, fx_t, fx):
    venue, kind, native = key
    _, mult = parse_base(inst["base"], venue)
    conv = lambda price, t: unit_price(price, mult, inst["quote"], fx[np.maximum(np.searchsorted(fx_t, t, side="right") - 1, 0)])
    qt = read("quotes", venue, days, [("native", "=", native), ("kind", "=", kind)], ["t", "bid", "ask"]).sort_values("t") if days else pd.DataFrame()
    t = qt["t"].to_numpy(float) if len(qt) else np.empty(0)
    bids = conv(qt["bid"].to_numpy(float), t) if len(qt) else np.empty(0)
    asks = conv(qt["ask"].to_numpy(float), t) if len(qt) else np.empty(0)
    bk = read("books", venue, days, [("native", "=", native), ("kind", "=", kind)])
    rows = []
    if not bk.empty:
        for r in bk.sort_values("t").itertuples():
            buy = vwap(list(zip(r.ap, r.aq)), q_usd)
            sell = vwap(list(zip(r.bp, r.bq)), q_usd)
            if buy is None or sell is None or not len(r.ap) or not len(r.bp):
                continue
            rows.append((r.t, buy, sell, (r.ap[0] + r.bp[0]) / 2))
    a = np.array(rows, float) if rows else np.empty((0, 4))
    bt = a[:, 0]
    mk = read("marks", venue, days, [("native", "=", native)], ["t", "mark"]) if kind == "perp" else pd.DataFrame()
    mk = mk.dropna().sort_values("t") if len(mk) else mk
    mt = mk["t"].to_numpy(float) if len(mk) else np.empty(0)
    ft, fr, _ = settled_funding(venue, native, days) if kind == "perp" else (np.empty(0), np.empty(0), False)
    return Leg(venue, kind, cfg["fees_bp"][venue][kind], bt, conv(a[:, 1], bt), conv(a[:, 2], bt), conv(a[:, 3], bt), t, bids, asks,
               mt, conv(mk["mark"].to_numpy(float), mt) if len(mk) else np.empty(0), ft, fr)


def load_pairs(cfg, days, q_usd):
    """Pairs that were candidates in these days, on these days' data only: nothing past the sample's end is read."""
    if not days:
        return [], 0
    t0, t1 = day_bounds(days[0])[0], day_bounds(days[-1])[1]
    eps = episodes()
    eps = eps[(eps["start"] < t1) & (eps["end"].isna() | (eps["end"] >= t0))]
    fx_t, fx = usdc_rate(cfg, days)
    insts = instruments()
    legs = {}
    pairs = []
    for (asset, ls, ss), _ in eps.groupby(["asset", "long", "short"]):
        kl, ks = tuple(ls.split("|")), tuple(ss.split("|"))
        if kl not in insts or ks not in insts:
            continue
        for k in (kl, ks):
            if k not in legs:
                legs[k] = build_leg(k, insts[k], cfg, days, q_usd, fx_t, fx)
        cat = category(asset, kl[0], kl[1], ks[0], ks[1], cfg["categories"])
        pairs.append(Pair(f"{ls}>{ss}", asset, cat, legs[kl], legs[ks]))
    return pairs, t1


def gaps_table():
    from .common import LOGS
    path = LOGS / "gaps.csv"
    if not path.exists():
        return pd.DataFrame(columns=["venue", "start", "end", "reason"])
    g = pd.read_csv(path)
    g["venue"] = g["venue"].str.split(":").str[0]
    g["start"] = pd.to_datetime(g["start_utc"]).astype("int64") // 10**9
    g["end"] = pd.to_datetime(g["end_utc"]).astype("int64") // 10**9
    return g[["venue", "start", "end", "reason"]]


def gap_share(days, min_venues=6):
    """Share of the sample's time with too few venues answering. Counted from the poll log, so it also covers a machine that was off."""
    if not days:
        return 1.0
    seen = collections.Counter()
    for f in (RAW / "snaps").glob("*/*"):
        if f.parent.name in days and f.suffix in (".parquet", ".shard"):
            venue = f.name.split(".")[0]
            for t in set(pq.read_table(f, columns=["t"])["t"].to_pylist()):
                seen[t // 60, venue] += 1
    per_min = collections.Counter(m for m, _ in seen)
    return 1 - sum(1 for n in per_min.values() if n >= min_venues) / (len(days) * 1440)
