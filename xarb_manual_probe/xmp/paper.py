"""Paper trades "as a human": the signal, then legs one by one after a delay, inside the owner's hours. All times are UTC seconds."""
from dataclasses import dataclass, field

import numpy as np

MSK = 3 * 3600


def vwap(levels, q_usd):
    """Average price of walking best-first (price, qty) levels for q_usd of notional; None when the book is too thin."""
    left, base = q_usd, 0.0
    for price, qty in levels:
        take = min(left, price * qty)
        base += take / price
        left -= take
        if left <= 1e-9:
            return q_usd / base
    return None


def spread_in(sell_short, buy_long, mid):
    return (sell_short - buy_long) / mid * 1e4


def spread_out(buy_short, sell_long, mid):
    return (buy_short - sell_long) / mid * 1e4


def in_hours(t, hours):
    h = ((t + MSK) % 86400) / 3600
    start, end = hours
    return start <= h < end or h < end - 24


def next_open(t, hours):
    """t itself in hours, else the next opening."""
    if in_hours(t, hours):
        return t
    h = ((t + MSK) % 86400) / 3600
    return t + (hours[0] - h) * 3600


@dataclass
class Leg:
    """One instrument, prices in USDT per unit of the asset, already at size Q."""
    venue: str
    kind: str
    fee_bp: float
    bt: np.ndarray            # book times and fills
    bbuy: np.ndarray
    bsell: np.ndarray
    bmid: np.ndarray
    qt: np.ndarray            # best prices
    qbid: np.ndarray
    qask: np.ndarray
    mt: np.ndarray = field(default_factory=lambda: np.empty(0))   # mark
    mark: np.ndarray = field(default_factory=lambda: np.empty(0))
    ft: np.ndarray = field(default_factory=lambda: np.empty(0))   # settled funding
    frate: np.ndarray = field(default_factory=lambda: np.empty(0))

    def book_at(self, t, stale):
        """Latest book not after t: (buy, sell, mid), NaN when older than `stale`. t may be an array."""
        t = np.asarray(t, float)
        i = np.searchsorted(self.bt, t, side="right") - 1
        ok = (i >= 0) & (t - self.bt[np.maximum(i, 0)] <= stale) if len(self.bt) else np.zeros(t.shape, bool)
        pick = lambda a: np.where(ok, a[np.maximum(i, 0)], np.nan) if len(self.bt) else np.full(t.shape, np.nan)
        return pick(self.bbuy), pick(self.bsell), pick(self.bmid)

    def quote_at(self, t, stale):
        t = np.asarray(t, float)
        i = np.searchsorted(self.qt, t, side="right") - 1
        ok = (i >= 0) & (t - self.qt[np.maximum(i, 0)] <= stale) if len(self.qt) else np.zeros(t.shape, bool)
        pick = lambda a: np.where(ok, a[np.maximum(i, 0)], np.nan) if len(self.qt) else np.full(t.shape, np.nan)
        b, a = pick(self.qbid), pick(self.qask)
        return a, b, (a + b) / 2

    def price_at(self, t, stale_book, stale_quote):
        """Book when fresh, else best prices (exits only: a book is written while the pair is a candidate)."""
        bb, bs, bm = self.book_at(t, stale_book)
        qb, qs, qm = self.quote_at(t, stale_quote)
        return np.where(np.isnan(bb), qb, bb), np.where(np.isnan(bs), qs, bs), np.where(np.isnan(bm), qm, bm)

    def funding_bp(self, sign, t_open, t_close, entry):
        """Accrued only while the leg is open at the settlement, on the notional of that moment. Positive rate: longs pay shorts."""
        if self.kind == "spot":
            return 0.0
        m = (self.ft > t_open) & (self.ft <= t_close)
        if not m.any():
            return 0.0
        px = self.quote_at(self.ft[m], 1e12)[2]
        px = np.where(np.isnan(px), entry, px)
        return float(np.sum(-sign * self.frate[m] * px / entry) * 1e4)

    def liquidation(self, sign, entry, t_from, t_to, lev, stale_quote):
        """First mark (mid where the venue gives none) that took 1/L against the leg. Spot has no margin call."""
        if self.kind == "spot":
            return None
        if len(self.mt):
            mask = (self.mt > t_from) & (self.mt <= t_to)
            t, px = self.mt[mask], self.mark[mask]
        else:
            mask = (self.qt > t_from) & (self.qt <= t_to)
            t, px = self.qt[mask], (self.qbid[mask] + self.qask[mask]) / 2
        hit = (px <= entry * (1 - 1 / lev)) if sign > 0 else (px >= entry * (1 + 1 / lev))
        return float(t[np.argmax(hit)]) if hit.any() else None


@dataclass
class Pair:
    pid: str
    asset: str
    cat: str
    long: Leg
    short: Leg
    flags: tuple = ()

    @property
    def fees_bp(self):
        return 2 * (self.long.fee_bp + self.short.fee_bp)

    def ticks(self, stale):
        """Book-based entry spread at the moments a new snapshot of either leg arrived. Past data only."""
        t = np.union1d(self.long.bt, self.short.bt)
        lb, _, lm = self.long.book_at(t, stale)
        _, ss, sm = self.short.book_at(t, stale)
        return t, spread_in(ss, lb, (lm + sm) / 2)

    def s_out(self, t, stale_book, stale_quote):
        _, ls, lm = self.long.price_at(t, stale_book, stale_quote)
        sb, _, sm = self.short.price_at(t, stale_book, stale_quote)
        return spread_out(sb, ls, (lm + sm) / 2)

    def s_in_best(self, t, stale):
        la, _, lm = self.long.quote_at(t, stale)
        _, sb, sm = self.short.quote_at(t, stale)
        return spread_in(sb, la, (lm + sm) / 2)

    def s_out_best(self, t, stale):
        _, lb, lm = self.long.quote_at(t, stale)
        sa, _, sm = self.short.quote_at(t, stale)
        return spread_out(sa, lb, (lm + sm) / 2)


@dataclass(frozen=True)
class Variant:
    theta_bp: float
    delay_s: float
    exit_name: str
    t_max_h: float
    tau_bp: float | None = None    # None: E2
    median_h: float = 24
    min_history_h: float = 12

    @property
    def name(self):
        return f"th{self.theta_bp:g}_D{self.delay_s:g}_{self.exit_name}_T{self.t_max_h:g}"


def variants(p, delays=None):
    out = []
    for th in p["thetas_bp"]:
        for d in delays if delays is not None else p["delays_s"]:
            for e in p["exits"]:
                for tm in p["t_max_h"]:
                    out.append(Variant(th, d, e["name"], tm, e.get("tau_bp"), e.get("median_h", 24), e.get("min_history_h", 12)))
    return out


def run_trade(pair, t_sig, v, p, data_end, transfer_bp=0.0, extra_delay=0.0):
    """One trade from a signal at t_sig. Reads prices only at the moments a person acts."""
    st, sq = p["book_stale_s"], p["quote_stale_s"]
    gap = p["second_leg_s"] if v.delay_s else 0
    t1 = t_sig + extra_delay + v.delay_s
    t2 = t1 + gap
    base = {"pid": pair.pid, "asset": pair.asset, "cat": pair.cat, "t_sig": t_sig, "t_in": t2, "long_venue": pair.long.venue, "short_venue": pair.short.venue}
    if t2 > data_end:
        return {**base, "status": "no_data"}
    buy, _, lm = (float(x) for x in pair.long.book_at(t1, st))
    b2, sell2, sm = (float(x) for x in pair.short.book_at(t2, st))
    _, sell1, sm1 = (float(x) for x in pair.short.book_at(t1, st))
    if np.isnan(buy) or np.isnan(sell2):
        return {**base, "status": "no_book"}
    if not np.isnan(sell1) and spread_in(sell1, buy, (lm + sm1) / 2) <= 0:
        return {**base, "status": "not_done"}
    s_in = spread_in(sell2, buy, (lm + sm) / 2)
    if s_in <= 0:
        return {**base, "status": "not_done"}

    tau = v.tau_bp
    if tau is None:
        hist = np.arange(t_sig - v.median_h * 3600, t_sig, 300.0)
        past = pair.s_out_best(hist, sq)
        past = past[~np.isnan(past)]
        if len(past) * 300 < v.min_history_h * 3600:
            return {**base, "status": "no_history"}
        tau = float(np.median(past))

    end = min(t2 + v.t_max_h * 3600, data_end)
    grid = np.arange(t2 + 15, end + 15, 15.0)
    so = pair.s_out(grid, st, sq)
    with np.errstate(invalid="ignore"):
        hit_t = np.nonzero(so <= tau)[0]
        hit_s = np.nonzero(so >= s_in + p["stop_bp"])[0]
    t_target = grid[hit_t[0]] if len(hit_t) else np.inf
    t_stop = grid[hit_s[0]] if len(hit_s) else np.inf
    lev = p["leverage"]
    liq_l = pair.long.liquidation(+1, buy, t1, end, lev, sq)
    liq_s = pair.short.liquidation(-1, sell2, t2, end, lev, sq)
    t_liq = min(x for x in (liq_l, liq_s, np.inf) if x is not None)
    t_trigger = min(t_target, t_stop, t_liq, end)
    reason = "liquidation" if t_trigger == t_liq else "stop" if t_trigger == t_stop else "target" if t_trigger == t_target else "t_max"

    te = next_open(t_trigger, p["hours_msk"]) if p["hours_apply_to_exits"] else t_trigger
    e1 = te + v.delay_s
    e2 = e1 + gap
    stale_exit = e2 > data_end
    # a book or quote may be missing at that moment: walk forward as a person would, within the data
    b, s, lm2, sm2 = np.nan, np.nan, np.nan, np.nan
    for shift in np.arange(0, 6 * 3600 + 1, 60.0):
        _, s, lm2 = (float(x) for x in pair.long.price_at(min(e1 + shift, data_end), st, sq))
        b, _, sm2 = (float(x) for x in pair.short.price_at(min(e2 + shift, data_end), st, sq))
        if not (np.isnan(b) or np.isnan(s)):
            e1, e2 = min(e1 + shift, data_end), min(e2 + shift, data_end)
            break
    if np.isnan(b) or np.isnan(s):  # end of data: the last prices there are, flagged
        s = float(pair.long.quote_at(e1, 1e12)[1])
        b = float(pair.short.quote_at(e2, 1e12)[0])
        lm2, sm2, stale_exit = s, b, True

    funding = pair.long.funding_bp(+1, t1, e1, buy) + pair.short.funding_bp(-1, t2, e2, sell2)
    if reason == "liquidation":
        liq_bp = -1e4 / lev  # the leg loses its margin and no more
        if t_liq == liq_l:
            price = liq_bp + (sell2 - b) / sell2 * 1e4
        else:
            price = liq_bp + (s - buy) / buy * 1e4
        s_out = float("nan")
    else:
        s_out = spread_out(b, s, (lm2 + sm2) / 2)
        price = s_in - s_out
    liq3_l = pair.long.liquidation(+1, buy, t1, e1, p["leverage_ref"], sq)
    liq3_s = pair.short.liquidation(-1, sell2, t2, e2, p["leverage_ref"], sq)
    net = price - pair.fees_bp + funding - transfer_bp
    return {**base, "status": "done", "t_out": max(e1, e2), "reason": reason, "s_in": s_in, "s_out": s_out, "price_bp": price, "fees_bp": pair.fees_bp, "funding_bp": funding,
            "transfer_bp": transfer_bp, "net_bp": net, "tau": tau, "stale_exit": stale_exit, "liq_ref": liq3_l is not None or liq3_s is not None}


class Margin:
    """Capital spread evenly over the venues, idle money included. A shortfall is a paid, delayed transfer from the richest venue not in use."""

    def __init__(self, capital, venues):
        self.bal = {v: capital / len(venues) for v in venues}

    def need(self, needs):
        """(transfers, feasible) for {venue: amount}."""
        trial = dict(self.bal)
        moved = 0
        for v, m in needs.items():
            short = m - trial[v]
            if short > 1e-9:
                src = max((x for x in trial if x not in needs), key=trial.get, default=None)
                if src is None or trial[src] < short:
                    return 0, False
                trial[src] -= short
                trial[v] += short
                moved += 1
        self.trial = trial
        return moved, True

    def commit(self, needs):
        self.bal = self.trial
        for v, m in needs.items():
            self.bal[v] -= m

    def release(self, needs, pnl):
        for v, m in needs.items():
            self.bal[v] += m + pnl / len(needs)


def margin_needs(pair, q_usd, lev):
    """Perp leg: Q/L. A spot long is bought in full: spot has no leverage."""
    needs: dict[str, float] = {}
    for leg in (pair.long, pair.short):
        needs[leg.venue] = needs.get(leg.venue, 0.0) + (q_usd if leg.kind == "spot" else q_usd / lev)
    return needs


def simulate(pairs, v, p, q_usd, capital, venues, data_end, rank=False):
    """Walks every tick above the threshold in time order; a run above the threshold gives at most one trade."""
    theta_t, theta_p, theta_r = [], [], []
    for k, pair in enumerate(pairs):
        if pair.long.venue not in venues or pair.short.venue not in venues:
            continue
        t, s = pair.ticks(p["book_stale_s"])
        with np.errstate(invalid="ignore"):
            above = s - pair.fees_bp >= v.theta_bp
        run = np.cumsum(~above)
        theta_t.append(t[above])
        theta_p.append(np.full(above.sum(), k))
        theta_r.append(run[above])
    if not theta_t:
        return []
    t_all = np.concatenate(theta_t)
    order = np.argsort(t_all, kind="stable")
    t_all, p_all, r_all = t_all[order], np.concatenate(theta_p)[order], np.concatenate(theta_r)[order]

    book = Margin(capital, venues)
    open_pos: list[dict] = []
    taken: dict[int, int] = {}
    busy = set()
    out = []
    for t, k, run in zip(t_all, p_all, r_all):
        k, run = int(k), int(run)
        for pos in [x for x in open_pos if x["t_out"] <= t]:
            open_pos.remove(pos)
            busy.discard(pos["k"])
            book.release(pos["needs"], pos["net_usd"])
        if k in busy or taken.get(k) == run or not in_hours(t, p["hours_msk"]) or len(open_pos) >= p["max_positions"]:
            continue
        pair = pairs[k]
        needs = margin_needs(pair, q_usd, p["leverage"])
        moved, ok = book.need(needs)
        taken[k] = run
        if not ok:
            out.append({"pid": pair.pid, "cat": pair.cat, "t_sig": float(t), "status": "no_capital"})
            continue
        tr = run_trade(pair, float(t), v, p, data_end, moved * p["transfer_usd"] / q_usd * 1e4, p["transfer_delay_s"] if moved else 0.0)
        if rank and tr["status"] == "done":
            spreads = [x.s_in_best(t, p["quote_stale_s"]) for x in pairs]
            tr["rank"] = 1 + sum(1 for x in spreads if x > pair.s_in_best(t, p["quote_stale_s"]))
        if tr["status"] == "done":
            tr["net_usd"] = tr["net_bp"] / 1e4 * q_usd
            tr["q_usd"] = q_usd
            tr["k"] = k
            book.commit(needs)
            open_pos.append({**tr, "needs": needs})
            busy.add(k)
        out.append(tr)
    return out
