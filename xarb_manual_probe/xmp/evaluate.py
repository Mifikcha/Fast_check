"""Metrics, the choice on development, the one control pass and the verdict."""
import json

import numpy as np
import pandas as pd

from . import paper, store
from .common import DERIVED, HOLDOUT, SELECTED, attempt, now_utc, require_frozen

CATS = ["K-SYN", "K-SP", "K-PP", "K-DEX"]


def day_of_trade(tr):
    return (tr["t_in"] // 86400).astype(int)


def block_ci(tr, level, n, seed):
    """Block bootstrap of the mean net result by day: trades of one day move together."""
    if tr.empty:
        return float("nan"), float("nan")
    days = day_of_trade(tr)
    groups = [g["net_bp"].to_numpy() for _, g in tr.groupby(days)]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(groups), (n, len(groups)))
    sums = np.array([g.sum() for g in groups])
    sizes = np.array([len(g) for g in groups])
    means = sums[idx].sum(1) / sizes[idx].sum(1)
    a = (1 - level) / 2
    return float(np.quantile(means, a)), float(np.quantile(means, 1 - a))


def without_top(tr, n=3):
    return tr["net_bp"].sort_values().iloc[:-n].sum() if len(tr) > n else float("nan")


def summary(rows, capital, n_days, cr):
    all_ = pd.DataFrame(rows)
    if all_.empty or "status" not in all_:
        return {"trades": 0, "not_done_share": float("nan"), "apr": float("nan")}
    tr = all_[all_["status"] == "done"].copy()
    tried = all_["status"].isin(["done", "not_done"]).sum()
    out = {"trades": len(tr), "signals": len(all_), "not_done_share": float((all_["status"] == "not_done").sum() / tried) if tried else float("nan"),
           "no_book": int((all_["status"] == "no_book").sum()), "no_capital": int((all_["status"] == "no_capital").sum())}
    if tr.empty:
        return {**out, "apr": 0.0}
    eq = tr.sort_values("t_out")["net_usd"].cumsum()
    lo, hi = block_ci(tr, cr["a3_ci"], cr["bootstrap_n"], cr["bootstrap_seed"])
    pair_sum = tr.groupby("pid")["net_bp"].sum().sort_values()
    out |= {"mean_bp": tr["net_bp"].mean(), "median_bp": tr["net_bp"].median(), "ci_lo": lo, "ci_hi": hi, "win_share": (tr["net_bp"] > 0).mean(), "worst_bp": tr["net_bp"].min(),
            "worst_of_capital": (tr["net_usd"] / capital).min(), "usd": tr["net_usd"].sum(), "apr": tr["net_usd"].sum() / capital * 365 / n_days,
            "max_dd_of_capital": float((eq.cummax().clip(lower=0) - eq).max() / capital), "price_bp": tr["price_bp"].mean(), "fees_bp": tr["fees_bp"].mean(),
            "funding_bp": tr["funding_bp"].mean(), "transfer_bp": tr["transfer_bp"].mean(), "sum_wo_top3": without_top(tr), "sum_wo_top3_pairs": pair_sum.iloc[:-3].sum() if len(pair_sum) > 3 else float("nan"),
            "liquidations": int((tr["reason"] == "liquidation").sum()), "liq_ref": int(tr["liq_ref"].sum()), "stale_exits": int(tr["stale_exit"].sum())}
    return out


def venue_sets(cfg):
    every = sorted(cfg["fees_bp"])
    return {"all": every, "owner": cfg["paper"]["owner_venues"] or every}


def load_all(cfg, days):
    return {q: store.load_pairs(cfg, days, q) for q in cfg["paper"]["q_usd"]}


def dev(cfg):
    """All 54 variants plus the robot reference; the choice rests on D = 60 s only."""
    require_frozen()
    p, cr = cfg["paper"], cfg["criteria"]
    days, _ = store.segments(cfg)
    if len(days) < cfg["samples"]["dev_days"]:
        raise SystemExit(f"development needs {cfg['samples']['dev_days']} full days, there are {len(days)}")
    data = load_all(cfg, days)
    pairs, end = data[p["main_q_usd"]]
    rows = []
    for v in paper.variants(p, p["delays_s"] + [p["robot_delay_s"]]):
        r = paper.simulate(pairs, v, p, p["main_q_usd"], p["main_capital_usd"], venue_sets(cfg)["all"], end)
        rows.append({"variant": v.name, "theta": v.theta_bp, "delay": v.delay_s, "exit": v.exit_name, "t_max": v.t_max_h, **summary(r, p["main_capital_usd"], len(days), cr)})
    t = pd.DataFrame(rows)
    DERIVED.mkdir(parents=True, exist_ok=True)
    t.to_csv(DERIVED.parent.parent / "reports" / "X2_dev_grid.csv", index=False)
    main = t[(t["delay"] == p["main_delay_s"]) & (t["trades"] >= p["min_dev_trades"])]
    best = main.sort_values(["apr", "theta"], ascending=[False, False]).iloc[0] if len(main) else t[t["delay"] == p["main_delay_s"]].sort_values(["trades", "theta"], ascending=[False, False]).iloc[0]
    sel = {"theta_bp": float(best["theta"]), "exit": best["exit"], "t_max_h": float(best["t_max"]), "dev_days": days, "dev_apr": float(best["apr"]), "dev_trades": int(best["trades"]), "utc": now_utc()}
    SELECTED.write_text(json.dumps(sel, indent=1), encoding="utf-8")
    attempt("dev", f"chosen {best['variant']}", require_frozen())
    return sel


def chosen(cfg, delay):
    sel = json.loads(SELECTED.read_text(encoding="utf-8"))
    e = next(x for x in cfg["paper"]["exits"] if x["name"] == sel["exit"])
    return paper.Variant(sel["theta_bp"], delay, e["name"], sel["t_max_h"], e.get("tau_bp"), e.get("median_h", 24), e.get("min_history_h", 12))


def open_holdout(cfg):
    """Once only, after the choice, on at least min_days of data."""
    digest = require_frozen()
    if HOLDOUT.exists():
        raise SystemExit(f"holdout already opened:\n{HOLDOUT.read_text(encoding='utf-8')}")
    if not SELECTED.exists():
        raise SystemExit("no chosen variant: run dev first")
    dev_days, ctl = store.segments(cfg)
    if len(dev_days) + len(ctl) < cfg["samples"]["min_days"] or len(ctl) < 7:
        raise SystemExit(f"recording too short: {len(dev_days)} + {len(ctl)} days")
    HOLDOUT.write_text(json.dumps({"utc": now_utc(), "section_sha256": digest, "days": ctl, "selected": json.loads(SELECTED.read_text(encoding="utf-8"))}, indent=1), encoding="utf-8")
    attempt("open-holdout", f"{len(ctl)} days", digest)
    return ctl


def holdout(cfg):
    """The single control pass of the chosen variant: main cell, delays, capital ladder, venue sets."""
    if not HOLDOUT.exists():
        raise SystemExit("holdout is closed: open-holdout first")
    res = DERIVED / "holdout_trades.parquet"
    if res.exists():
        raise SystemExit("control already computed: variants are not re-run after it")
    require_frozen()
    p, cr = cfg["paper"], cfg["criteria"]
    days = json.loads(HOLDOUT.read_text(encoding="utf-8"))["days"]
    data = load_all(cfg, days)
    rows = []
    for q, (pairs, end) in data.items():
        for delay in [p["robot_delay_s"], *p["delays_s"]]:
            v = chosen(cfg, delay)
            for capital in p["capital_usd"]:
                for name, vs in venue_sets(cfg).items():
                    if name == "owner" and not p["owner_venues"]:
                        continue  # no answer yet: the owner's set is the full one
                    main = q == p["main_q_usd"] and delay == p["main_delay_s"] and capital == p["main_capital_usd"]
                    rows += [{**r, "q": q, "delay": delay, "capital": capital, "venues": name} for r in paper.simulate(pairs, v, p, q, capital, vs, end, rank=main)]
    t = pd.DataFrame(rows)
    t.to_parquet(res)
    attempt("holdout", f"{len(t)} rows", require_frozen())
    return t


def criteria_table(cell, cat_rows, cfg, n_days, gap, venues_alive):
    cr = cfg["criteria"]
    s = summary(cell.to_dict("records"), cfg["paper"]["main_capital_usd"], n_days, cr)
    a1 = n_days + cfg["samples"]["dev_days"] >= cr["a1_min_days"] and gap <= cr["a1_max_gap_share"] and venues_alive >= cr["a1_min_venues"]
    a2 = s.get("apr", 0) >= cr["a2_min_apr"]
    enough = s.get("trades", 0) >= cr["a3_min_trades"]
    a3 = enough and s.get("ci_lo", -1) > 0 and s.get("sum_wo_top3", -1) > 0
    a4 = s.get("liquidations", 0) <= cr["a4_max_liquidations"] and s.get("worst_of_capital", 0) >= cr["a4_worst_trade_of_capital"]
    verdict = "мало данных" if not a1 or not enough else "бумажно выгодно" if a2 and a3 and a4 else "бумажно невыгодно"
    return s, {"A1": a1, "A2": a2, "A3": a3, "A4": a4}, verdict


def category_verdicts(cell, cfg, n_days, gap, venues_alive):
    out = {}
    for cat in CATS:
        part = cell[cell["cat"] == cat]
        s, marks, v = criteria_table(part, None, cfg, n_days, gap, venues_alive)
        if s.get("trades", 0) < cfg["criteria"]["category_min_trades"]:
            v = "мало данных"
        out[cat] = (s, marks, v)
    return out


def winner_curse(tr, n=10):
    tr = tr[tr["status"] == "done"]
    if len(tr) < n:
        return pd.DataFrame()
    d = pd.qcut(tr["s_in"], n, labels=False, duplicates="drop")
    return tr.groupby(d).agg(s_in=("s_in", "mean"), net_bp=("net_bp", "mean"), trades=("net_bp", "size")).reset_index(drop=True)


def convergence(cfg, days):
    """Episodes by whether s_out reached 10 bp within 1, 24, 72 hours; descriptive, needs no trades."""
    out = []
    pairs, end = store.load_pairs(cfg, days, cfg["paper"]["main_q_usd"])
    by_id = {p.pid: p for p in pairs}
    eps = store.episodes()
    t0, t1 = store.day_bounds(days[0])[0], store.day_bounds(days[-1])[1]
    for e in eps[(eps["start"] >= t0) & (eps["start"] < t1)].itertuples():
        p = by_id.get(f"{e.long}>{e.short}")
        if p is None:
            continue
        g = np.arange(e.start, min(e.start + 72 * 3600, end), 60.0)
        so = p.s_out_best(g, cfg["paper"]["quote_stale_s"])
        with np.errstate(invalid="ignore"):
            hit = np.nonzero(so <= 10)[0]
        h = (g[hit[0]] - e.start) / 3600 if len(hit) else np.inf
        out.append({"cat": p.cat, "pid": p.pid, "start": e.start, "dur_h": ((e.end if e.end == e.end else t1) - e.start) / 3600, "s_in_bp": e.s_in_bp, "conv_h": h})
    return pd.DataFrame(out)
