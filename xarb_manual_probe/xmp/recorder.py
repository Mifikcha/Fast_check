"""Background market recorder. Public data only. One thread per job per venue; a stuck venue never blocks the others."""
import collections
import csv
import datetime as dt
import json
import logging
import shutil
import socket
import struct
import threading
import time
from dataclasses import asdict

import pyarrow as pa
import pyarrow.parquet as pq

from .common import LOGS, ROOT
from .match import can_pair, pair_ok, parse_base, unit_price
from .venues import VENUES, Inst, RateLimited

RAW = ROOT / "data" / "raw"
log = logging.getLogger("x2")

F32 = pa.float32()
F64 = pa.float64()
S = pa.dictionary(pa.int32(), pa.string())
SCHEMAS = {
    "quotes": pa.schema([("kind", S), ("native", S), ("t", pa.int32()), ("bid", F32), ("ask", F32), ("bq", F32), ("aq", F32), ("ex_lag_ms", pa.int32())]),
    "snaps": pa.schema([("kind", S), ("t", pa.int32()), ("n", pa.int32()), ("ms", pa.int32())]),
    "marks": pa.schema([("native", S), ("t", pa.int32()), ("mark", F32), ("index", F32)]),
    "funding": pa.schema([("native", S), ("t", pa.int32()), ("rate", F64), ("next_ms", pa.int64()), ("interval_h", F32)]),
    "volume": pa.schema([("native", S), ("t", pa.int32()), ("vol", F64), ("oi", F64)]),
    "wallet": pa.schema([("coin", S), ("t", pa.int32()), ("dep", pa.bool_()), ("wd", pa.bool_())]),
    "fund_hist": pa.schema([("native", S), ("ms", pa.int64()), ("rate", F64), ("t", pa.int32())]),
    "books": pa.schema([("kind", S), ("native", S), ("t", pa.int32()), ("bp", pa.list_(F64)), ("bq", pa.list_(F32)), ("ap", pa.list_(F64)), ("aq", pa.list_(F32))]),
}
SORT = {"quotes": ["kind", "native", "t"], "snaps": ["kind", "t"], "marks": ["native", "t"], "funding": ["native", "t"], "volume": ["native", "t"],
        "wallet": ["coin", "t"], "fund_hist": ["native", "ms"], "books": ["kind", "native", "t"]}
FLOATS = {"bid", "ask", "bq", "aq", "mark", "index"}


def plain_schema(table):
    return pa.schema([pa.field(f.name, pa.string() if pa.types.is_dictionary(f.type) else f.type) for f in SCHEMAS[table]])


def day_of(t):
    return dt.datetime.fromtimestamp(t, dt.timezone.utc).strftime("%Y-%m-%d")


def write_parquet(tb, path):
    split = [c for c in tb.column_names if c in FLOATS and not pa.types.is_list(tb.schema.field(c).type)]
    pq.write_table(tb, path, compression="zstd", compression_level=9, use_dictionary=[c for c in tb.column_names if c not in split], use_byte_stream_split=split)


class Sink:
    """Rows wait in memory, reach disk as 10-minute shards; yesterday's shards merge into one sorted file, which compresses far better."""

    def __init__(self, raw=RAW):
        self.raw = raw
        self.rows = collections.defaultdict(list)
        self.lock = threading.Lock()

    def add(self, table, venue, row):
        with self.lock:
            self.rows[table, venue].append(row)

    def flush(self):
        with self.lock:
            rows, self.rows = self.rows, collections.defaultdict(list)
        for (table, venue), rs in rows.items():
            try:
                self.write_shards(table, venue, rs)
            except Exception:
                log.exception("flush %s %s, %d rows lost", table, venue, len(rs))  # one bad value must not take the other tables with it

    def write_shards(self, table, venue, rs):
        cols = list(zip(*rs))
        tb = pa.table([pa.array(c, type=f.type) for c, f in zip(cols, SCHEMAS[table])], schema=SCHEMAS[table])
        by_day = collections.defaultdict(list)
        for k, t in enumerate(tb["t"].to_pylist()):
            by_day[day_of(t)].append(k)
        for d, idx in by_day.items():
            folder = self.raw / table / d
            folder.mkdir(parents=True, exist_ok=True)
            pq.write_table(tb.take(idx), folder / f"{venue}.{time.time_ns()}.shard", compression="zstd")

    def compact(self, today):
        for folder in sorted(self.raw.glob("*/*")):
            if folder.name >= today or not folder.is_dir():
                continue
            shards = collections.defaultdict(list)
            for f in folder.glob("*.shard"):
                shards[f.name.split(".")[0]].append(f)
            table = folder.parent.name
            for venue, fs in shards.items():
                final = folder / f"{venue}.parquet"
                old = [final] if final.exists() else []
                tb = pa.concat_tables([pq.read_table(f).cast(plain_schema(table)) for f in old + fs]).sort_by([(c, "ascending") for c in SORT[table]])  # no sort on dictionaries
                tmp = final.with_suffix(".tmp")
                write_parquet(tb, tmp)
                tmp.replace(final)
                for f in fs:
                    f.unlink()


def sntp_offset(host="pool.ntp.org"):
    """Local clock minus NTP, seconds."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.settimeout(5)
    t0 = time.time()
    s.sendto(b"\x1b" + 47 * b"\0", (host, 123))
    d = s.recv(48)
    t1 = time.time()
    s.close()
    sec, frac = struct.unpack("!II", d[40:48])
    return (t0 + t1) / 2 - (sec - 2208988800 + frac / 2**32)


def trim_book(side, ct, need_usd):
    """Levels until `need_usd` is covered (plus one), sizes in base units: enough for any Q of the stage."""
    p, q, total = [], [], 0.0
    for price, qty in side:
        p.append(price)
        q.append(qty * ct)
        total += price * qty * ct
        if total >= need_usd:
            break
    return p, q


class Recorder:
    def __init__(self, cfg):
        self.cfg = cfg
        self.rc = cfg["recording"]
        self.sink = Sink()
        self.stop = threading.Event()
        self.insts: dict[tuple, Inst] = {}
        self.latest: dict[tuple, tuple] = {}
        self.sent: dict[tuple, tuple] = {}
        self.vol: dict[tuple, float] = {}
        self.samples = collections.defaultdict(lambda: collections.deque(maxlen=24))
        self.cand: dict[tuple, dict] = {}
        self.recent_legs: dict[tuple, float] = {}
        self.hot: set[tuple] = set()
        self.gap_open: dict[str, tuple] = {}
        self.lock = threading.Lock()

    def gap(self, venue, reason=None):
        """Open a gap on the first failure, close it with the first success."""
        now = time.time()
        with self.lock:
            if reason and venue not in self.gap_open:
                self.gap_open[venue] = (now, reason)
                log.warning("%s: %s", venue, reason)
            elif not reason and venue in self.gap_open:
                start, why = self.gap_open.pop(venue)
                self.gap_row(venue, start, now, why)

    @staticmethod
    def gap_row(venue, start, end, why):
        path = LOGS / "gaps.csv"
        new = not path.exists()
        with path.open("a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["venue", "start_utc", "end_utc", "reason"])
            w.writerow([venue, dt.datetime.fromtimestamp(start, dt.timezone.utc).isoformat(timespec="seconds"), dt.datetime.fromtimestamp(end, dt.timezone.utc).isoformat(timespec="seconds"), why])

    def guarded(self, venue, fn, what):
        """Run one poll; 429 waits as told, any other failure is a gap, never a crash."""
        try:
            out = fn()
            self.gap(f"{venue}:{what}")
            return out
        except RateLimited as e:
            self.gap(f"{venue}:{what}", "429")
            self.stop.wait(e.retry_after)
        except Exception as e:
            self.gap(f"{venue}:{what}", repr(e)[:120])
        return None

    def grid(self, period, phase=0.0):
        while not self.stop.is_set():
            now = time.time()
            nxt = (now - phase) // period * period + period + phase
            if self.stop.wait(nxt - now):
                return
            yield int(nxt)

    def put_quotes(self, venue, kind, rows, t):
        known = 0
        for native, bid, ask, bq, aq, ex in rows:
            inst = self.insts.get((venue, kind, native))
            if not inst or not bid or not ask or bid >= ask * 1.5:
                continue
            known += 1
            leg = (venue, kind, native)
            lag = max(int(ex - t * 1000), -2_000_000_000) if ex else None  # dead markets carry months-old times
            self.latest[leg] = (bid, ask, t, lag)
            prev = self.sent.get(leg)
            if prev is None or (bid, ask) != prev[:2] or t - prev[2] >= 3600:  # disk budget: unchanged prices are implied, sizes then lag at most an hour
                self.sent[leg] = (bid, ask, t)
                self.sink.add("quotes", venue, (kind, native, t, bid, ask, bq * inst.ct if bq else None, aq * inst.ct if aq else None, lag))
        return known

    def poll_quotes(self, venue):
        v = VENUES[venue]
        for t in self.grid(self.rc["quote_s"]):
            for kind in v.kinds:
                t0 = time.time()
                rows = self.guarded(venue, lambda k=kind: v.tickers(k), f"quotes_{kind}")
                if rows is not None:
                    n = self.put_quotes(venue, kind, rows, t)
                    self.sink.add("snaps", venue, (kind, t, n, int((time.time() - t0) * 1000)))

    def run_stream(self, venue):
        v = VENUES[venue]
        pool: dict[str, tuple] = {}
        key = f"{venue}:stream"

        def put(native, bid, ask, bq, aq, ex):
            pool[native] = (native, bid, ask, bq, aq, ex)
            if key in self.gap_open:  # the call only returns when the socket dies: the first message ends the gap
                self.gap(key)

        def snapper():
            for t in self.grid(self.rc["quote_s"]):
                if pool and key not in self.gap_open:
                    n = self.put_quotes(venue, "perp", list(pool.values()), t)
                    self.sink.add("snaps", venue, ("perp", t, n, 0))

        threading.Thread(target=snapper, daemon=True).start()

        def socket_loop(chunk):
            while not self.stop.is_set():
                self.guarded(venue, lambda: v.stream(chunk, put, self.stop.is_set), "stream")
                self.gap(key, "socket closed")
                self.stop.wait(5)

        while not self.stop.is_set() and not any(k[0] == venue for k in self.insts):
            self.stop.wait(1)
        # shortcut: instruments listed later wait for a restart, the scheduler brings one after any reboot
        insts = [i for k, i in list(self.insts.items()) if k[0] == venue]
        for k in range(0, len(insts), v.socket_size):
            threading.Thread(target=socket_loop, args=(insts[k:k + v.socket_size],), daemon=True).start()
        self.stop.wait()

    def load_insts(self, venue):
        rows = self.guarded(venue, VENUES[venue].insts, "instruments")
        if rows:
            for i in rows:
                self.insts[venue, i.kind, i.native] = i
            d = LOGS.parent / "data" / "raw" / "insts"
            d.mkdir(parents=True, exist_ok=True)
            (d / f"{venue}.{day_of(time.time())}.json").write_text(json.dumps([asdict(i) for i in rows]), encoding="utf-8")

    def poll_stats(self, venue):
        v = VENUES[venue]
        self.load_insts(venue)
        for t in self.grid(60, phase=7):
            if t % 3600 < 60:
                self.load_insts(venue)
            s = self.guarded(venue, v.stats, "stats")
            if not s:
                continue
            for native, r in s.items():
                if (venue, "perp", native) not in self.insts:
                    continue
                if r.get("mark") or r.get("index"):
                    self.sink.add("marks", venue, (native, t, r.get("mark"), r.get("index")))
                if t % self.rc["funding_s"] < 60 and r.get("funding") is not None:
                    self.sink.add("funding", venue, (native, t, r["funding"], int(r["next_ms"]) if r.get("next_ms") else None, r.get("interval_h")))
                if r.get("vol") is not None:
                    self.vol[venue, "perp", native] = r["vol"]
                if t % 3600 < 60:
                    self.sink.add("volume", venue, (native, t, r.get("vol"), r.get("oi")))
            if v.wallet and t % 3600 < 60:
                for coin, dep, wd in self.guarded(venue, v.wallet, "wallet") or []:
                    self.sink.add("wallet", venue, (coin, t, bool(dep), bool(wd)))

    def scan_once(self, t):
        """Candidates by best prices (section 4). Book polling follows this set."""
        rc = self.rc
        fx = self.latest.get(tuple(rc["usdc_leg"]))
        rate = (fx[0] + fx[1]) / 2 if fx and t - fx[2] <= rc["quote_stale_s"] else 1.0
        groups = collections.defaultdict(list)
        for leg, (bid, ask, seen, lag) in list(self.latest.items()):
            if t - seen > rc["quote_stale_s"] or (lag is not None and -lag > rc["exchange_stale_s"] * 1000):
                continue
            i = self.insts[leg]
            asset, mult = parse_base(i.base, i.venue)
            groups[asset].append((leg, unit_price(bid, mult, i.quote, rate), unit_price(ask, mult, i.quote, rate)))
        hourly = t % 3600 < rc["quote_s"]
        seen_keys = set()
        hot = set()
        for asset, legs in groups.items():
            if len(legs) < 2:
                continue
            for la, ba, aa in legs:
                for ls, bs, as_ in legs:
                    if la == ls or not can_pair(la[1], ls[1]):
                        continue
                    key = (la, ls)
                    mid = (ba + aa + bs + as_) / 4
                    s_in = (bs - aa) / mid * 1e4
                    if hourly or key not in self.samples:
                        self.samples[key].append(((ba + aa) - (bs + as_)) / (bs + as_))
                    c = self.cand.get(key)
                    if c:
                        seen_keys.add(key)
                        if s_in >= rc["cand_on_bp"]:
                            hot |= set(key)
                        if s_in >= rc["cand_off_bp"]:
                            c["below"] = None
                        elif c["below"] is None:
                            c["below"] = t
                        elif t - c["below"] >= rc["cand_off_s"]:
                            del self.cand[key]
                            self.cand_event(t, "end", asset, key, s_in)
                    elif s_in >= rc["cand_on_bp"] and pair_ok(self.samples[key], rc["unit_check"]):
                        self.cand[key] = {"start": t, "below": None}
                        self.cand_event(t, "start", asset, key, s_in)
                        seen_keys.add(key)
                        hot |= set(key)
        self.hot = hot
        for key in seen_keys:
            self.recent_legs[key[0]] = self.recent_legs[key[1]] = t

    @staticmethod
    def cand_event(t, ev, asset, key, s_in):
        path = RAW / "candidates.csv"
        new = not path.exists()
        with path.open("a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["t", "event", "asset", "long", "short", "s_in_bp"])
            w.writerow([t, ev, asset, "|".join(key[0]), "|".join(key[1]), round(s_in, 2)])

    def restore_candidates(self):
        """An episode survives a restart: it has no `end` row yet."""
        path = RAW / "candidates.csv"
        if not path.exists():
            return
        with path.open(encoding="utf-8") as f:
            for r in csv.DictReader(f):
                key = (tuple(r["long"].split("|")), tuple(r["short"].split("|")))
                if r["event"] == "start":
                    self.cand[key] = {"start": int(r["t"]), "below": None}
                else:
                    self.cand.pop(key, None)

    def scanner(self):
        for t in self.grid(self.rc["quote_s"], phase=9):
            try:
                self.scan_once(t)
            except Exception:
                log.exception("scan")

    def poll_books(self, venue, last):
        """One of several workers of a venue: requests wait on the network, not on the rate limit, and the workers share `book_rps`."""
        v = VENUES[venue]
        pause = self.rc["book_workers"] / self.rc["book_rps"][venue]
        while not self.stop.is_set():
            with self.lock:
                legs = {leg for key in list(self.cand) for leg in key if leg[0] == venue}
                # pairs above the entry level first: their turn must stay near book_s, the waiting rest takes what is left
                due = [leg for leg in legs if time.time() - last.get(leg, 0) >= self.rc["book_s"]]
                leg = min(due, key=lambda x: (x not in self.hot, last.get(x, 0)), default=None)
                if leg:
                    last[leg] = time.time()
            if not leg:
                self.stop.wait(0.5)
                continue
            t = int(time.time())
            got = self.guarded(venue, lambda: v.book(leg[1], leg[2]), "book")
            i = self.insts.get(leg)
            if got and i:
                bp, bq = trim_book(got[0], i.ct, self.rc["book_usd"])
                ap, aq = trim_book(got[1], i.ct, self.rc["book_usd"])
                self.sink.add("books", venue, (leg[1], leg[2], t, bp, bq, ap, aq))
            self.stop.wait(pause)

    def poll_hist(self, venue):
        """Settled funding of the legs a paper trade may have held: candidates of the last 80 hours."""
        v = VENUES[venue]
        pause = 1 / self.rc["book_rps"][venue]
        for t in self.grid(3600, phase=300):
            legs = [leg for leg, seen in list(self.recent_legs.items()) if leg[0] == venue and leg[1] == "perp" and t - seen < 80 * 3600]
            for leg in legs:
                for ms, rate in self.guarded(venue, lambda: v.hist(leg[2]), "fund_hist") or []:
                    if t * 1000 - ms < 96 * 3_600_000:
                        self.sink.add("fund_hist", venue, (leg[2], ms, rate, t))
                self.stop.wait(pause)

    def house(self):
        """Flush, daily merge, heartbeat, clock, disk reserve."""
        hb = LOGS / "heartbeat.txt"
        for t in self.grid(60):
            hb.write_text(str(t), encoding="utf-8")
            if t % self.rc["flush_s"] < 60:
                log.info("candidates %d, legs above entry level %d, open gaps %d", len(self.cand), len(self.hot), len(self.gap_open))
                try:
                    self.sink.flush()
                    self.sink.compact(day_of(t))
                except Exception:
                    log.exception("flush")
            if t % 3600 < 60:
                try:
                    with (LOGS / "clock.csv").open("a", encoding="utf-8") as f:
                        f.write(f"{t},{sntp_offset():.3f}\n")
                except OSError as e:
                    log.warning("ntp: %r", e)
            free_gb = shutil.disk_usage(ROOT).free / 2**30
            if free_gb < self.rc["disk_reserve_gb"]:
                (LOGS / "STOPPED_DISK").write_text(f"{t}: free {free_gb:.1f} GB, reserve {self.rc['disk_reserve_gb']} GB\n", encoding="utf-8")
                log.error("disk reserve reached, stopping")
                self.stop.set()

    def keep(self, job):
        """A dead thread would silence a venue for weeks: log it and start again."""
        while not self.stop.is_set():
            try:
                job()
            except Exception:
                log.exception("job died")
                self.stop.wait(10)

    def note_downtime(self):
        hb = LOGS / "heartbeat.txt"
        if hb.exists():
            last = int(hb.read_text(encoding="utf-8"))
            if time.time() - last > 120:
                self.gap_row("ALL", last, time.time(), "recorder down")

    def run(self, seconds=None):
        LOGS.mkdir(exist_ok=True)
        RAW.mkdir(parents=True, exist_ok=True)
        self.note_downtime()
        self.restore_candidates()
        jobs = [self.house, self.scanner]
        for name, v in VENUES.items():
            jobs.append(lambda n=name: self.poll_stats(n))
            turns: dict[tuple, float] = {}
            jobs += [lambda n=name, last=turns: self.poll_books(n, last) for _ in range(self.rc["book_workers"])]
            jobs.append(lambda n=name: (self.poll_quotes if VENUES[n].tickers else self.run_stream)(n))
            if v.hist:
                jobs.append(lambda n=name: self.poll_hist(n))
        threads = [threading.Thread(target=self.keep, args=(j,), daemon=True) for j in jobs]
        for th in threads:
            th.start()
        self.stop.wait(seconds)
        self.stop.set()
        self.sink.flush()
        with self.lock:
            for venue, (start, why) in self.gap_open.items():
                self.gap_row(venue, start, time.time(), why)

