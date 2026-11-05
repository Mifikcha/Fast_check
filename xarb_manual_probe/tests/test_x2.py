"""Stage X2A: section 8 of the task. Everything runs on synthetic data, no network."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from xmp import common, evaluate, paper, recorder  # noqa: E402
from xmp.match import can_pair, category, pair_ok, parse_base, unit_price  # noqa: E402
from xmp.paper import Leg, Margin, Pair, Variant, in_hours, margin_needs, next_open, run_trade, simulate, spread_in, spread_out, vwap  # noqa: E402

P = {"book_stale_s": 30, "quote_stale_s": 3700, "second_leg_s": 20, "stop_bp": 300, "leverage": 2, "leverage_ref": 3, "max_positions": 1,
     "hours_msk": [10, 25], "hours_apply_to_exits": True, "transfer_usd": 1, "transfer_delay_s": 1800}
NOON_MSK = 1_800_000_000 - 1_800_000_000 % 86400 + 9 * 3600  # 12:00 MSK


def leg(venue, t, bid, ask, kind="perp", fee=5.0, **kw):
    """Book and best prices alike: flat depth around the given bid and ask."""
    t = np.asarray(t, float)
    bid, ask = np.broadcast_to(bid, t.shape).astype(float), np.broadcast_to(ask, t.shape).astype(float)
    return Leg(venue, kind, fee, t, ask, bid, (ask + bid) / 2, t, bid, ask, **kw)


def path(t, before, jumps):
    """Piecewise constant price: `before`, then value from each (time, value) on."""
    x = np.full(len(t), float(before))
    for at, val in jumps:
        x[t >= at] = val
    return x


def t_grid(hours=6, step=10):
    return NOON_MSK + np.arange(0, hours * 3600, step, dtype=float)


def pair_simple(long_ask=100.0, short_bid=101.0, close_at=None, t=None):
    """Long at 100, short at 101 (about 100 bp); from close_at the short falls to the long's price."""
    t = t_grid() if t is None else t
    s_bid = path(t, short_bid, [(close_at, 100.0)] if close_at else [])
    s_ask = s_bid + 0.01
    return Pair("a>b", "X", "K-PP", leg("a", t, long_ask - 0.01, long_ask), leg("b", t, s_bid, s_ask))


V = Variant(50, 60, "E1_10", 24, tau_bp=10)


def test_vwap_known_answer():
    book = [(100, 1), (101, 1), (110, 100)]
    assert vwap(book, 100) == pytest.approx(100)
    assert vwap(book, 201) == pytest.approx(201 / 2)
    assert vwap(book, 311) == pytest.approx(311 / 3)
    assert vwap([(100, 1)], 101) is None


def test_spreads_both_directions():
    bid_a, ask_a, bid_b, ask_b = 99.9, 100.0, 100.9, 101.0
    mid = (bid_a + ask_a + bid_b + ask_b) / 4
    assert spread_in(bid_b, ask_a, mid) == pytest.approx(0.9 / mid * 1e4)   # long a, short b
    assert spread_in(bid_a, ask_b, mid) == pytest.approx(-1.1 / mid * 1e4)  # long b, short a
    assert spread_out(ask_b, bid_a, mid) == pytest.approx(-spread_in(bid_a, ask_b, mid))


def test_decision_ignores_the_future():
    t = t_grid()
    a, b = pair_simple(t=t), pair_simple(t=t)
    cut = t[100]
    late = t > cut
    b.short.bsell[late] = 500.0
    b.short.bbuy[late] = 501.0
    ta, sa = a.ticks(30)
    tb, sb = b.ticks(30)
    m = ta <= cut
    assert np.array_equal(sa[m], sb[m])
    a_as_of, b_as_of = a.short.book_at(cut, 30), b.short.book_at(cut, 30)
    assert a_as_of == b_as_of


def test_entry_uses_book_at_delay_and_second_leg_later():
    t = t_grid()
    sig = t[10]
    d = 60
    short_bid = path(t, 101.0, [(sig + d, 102.0), (sig + d + 20, 103.0)])
    pair = Pair("a>b", "X", "K-PP", leg("a", t, path(t, 99.99, [(sig + d, 100.99)]), path(t, 100.0, [(sig + d, 101.0)])), leg("b", t, short_bid, short_bid + 0.01))
    r = run_trade(pair, sig, Variant(50, d, "E1_0", 1, tau_bp=-1e6), P, t[-1])
    assert r["status"] == "done"
    mid = (100.995 + 103.005) / 2
    assert r["s_in"] == pytest.approx((103 - 101) / mid * 1e4)  # bought at t + D (101), sold 20 s later (103), nothing from the signal moment


def test_exit_after_delay_legs_in_turn():
    t = t_grid()
    close = t[100]
    r = run_trade(pair_simple(close_at=close, t=t), t[10], V, P, t[-1])
    assert r["reason"] == "target"
    assert r["t_out"] >= close + 60 + 20
    assert r["t_out"] <= close + 15 + 60 + 20


def test_vanished_spread_is_not_done():
    t = t_grid()
    sig = t[10]
    pair = pair_simple(close_at=sig + 30, t=t)
    assert run_trade(pair, sig, V, P, t[-1])["status"] == "not_done"


def test_funding_sign_and_intervals():
    for hours in (1, 4, 8):
        ft = NOON_MSK + np.arange(1, 9) * 3600.0 * hours
        rate = np.full(len(ft), 0.001)
        t = np.array([NOON_MSK, NOON_MSK + 100 * 3600.0])
        lg = leg("a", t, 100, 100.1, ft=ft, frate=rate)
        close = NOON_MSK + 8 * hours * 3600.0
        long_bp = lg.funding_bp(+1, NOON_MSK, close, 100.05)
        assert long_bp == pytest.approx(-8 * 10.0, rel=1e-6)                       # longs pay a positive rate
        assert lg.funding_bp(-1, NOON_MSK, close, 100.05) == pytest.approx(80.0, rel=1e-6)   # shorts receive it
        assert lg.funding_bp(+1, ft[0], ft[1], 100.05) == pytest.approx(-10.0, rel=1e-6)     # opened right at a settlement: only the next counts
        assert lg.funding_bp(+1, ft[1] + 1, ft[2] - 1, 100.05) == 0.0                         # no settlement while open
    assert leg("a", t, 100, 100.1, kind="spot", ft=ft, frate=rate).funding_bp(+1, NOON_MSK, close, 100) == 0.0


def test_ticker_and_contract_unit():
    assert parse_base("1000PEPE") == ("PEPE", 1000)
    assert parse_base("PEPE") == ("PEPE", 1)
    assert parse_base("kPEPE", "hyperliquid") == ("PEPE", 1000)
    assert parse_base("KPEPE", "bybit") == ("KPEPE", 1)
    assert parse_base("US100") == ("US100", 1)
    assert parse_base("SHIB1000") == ("SHIB", 1000)
    assert parse_base("1MBABYDOGE") == ("BABYDOGE", 1_000_000)
    assert unit_price(0.012, 1000, "USDT", 1.0) == pytest.approx(0.000012)
    assert unit_price(100, 1, "USDC", 0.999) == pytest.approx(99.9)
    assert unit_price(100, 1, "USDT", 0.999) == 100
    assert pair_ok([0.001, 0.015, 0.03]) and not pair_ok([0.05, 0.04, 0.001]) and not pair_ok([])


def test_categories_and_spot_short():
    cat = {"syn_assets": ["OPENAI", "AAPL"], "syn_regex": "[A-Z]+[0-9]+IX", "dex": ["lighter"]}
    assert category("OPENAI", "lighter", "perp", "okx", "perp", cat) == "K-SYN"
    assert category("AAPLX", "okx", "spot", "okx", "perp", cat) == "K-SYN"
    assert category("UK100IX", "okx", "perp", "bybit", "perp", cat) == "K-SYN"
    assert category("BTC", "lighter", "perp", "okx", "perp", cat) == "K-DEX"
    assert category("BTC", "okx", "spot", "bybit", "perp", cat) == "K-SP"
    assert category("BTC", "okx", "perp", "bybit", "perp", cat) == "K-PP"
    assert can_pair("spot", "perp") and can_pair("perp", "perp") and not can_pair("perp", "spot")


def test_hours():
    h = [10, 25]
    day = NOON_MSK - 12 * 3600  # 00:00 MSK
    assert in_hours(day + 10 * 3600, h) and in_hours(day + 23.5 * 3600, h) and in_hours(day + 0.5 * 3600, h)
    assert not in_hours(day + 1.5 * 3600, h) and not in_hours(day + 9.9 * 3600, h)
    assert next_open(day + 3 * 3600, h) == pytest.approx(day + 10 * 3600)


def test_max_positions_and_hours_in_simulate():
    t = t_grid(hours=4)
    pairs = []
    for k in range(3):
        p = pair_simple(t=t)
        p.pid = f"p{k}"
        p.long = leg(f"a{k}", t, 99.99, 100.0)
        p.short = leg(f"b{k}", t, np.full(len(t), 101.0), 101.01)
        pairs.append(p)
    venues = sorted({x.long.venue for x in pairs} | {x.short.venue for x in pairs})
    done = [r for r in simulate(pairs, V, P, 1000, 60000, venues, t[-1]) if r["status"] == "done"]
    assert len(done) == 1  # M = 1 and the first trade lasts to the end of the data
    night = t_grid(hours=4) - 10 * 3600  # 02:00-06:00 MSK
    p = pair_simple(t=night)
    assert simulate([p], V, P, 1000, 60000, ["a", "b"], night[-1]) == []


def test_liquidation_closes_both_legs_margin_not_negative():
    t = t_grid()
    sig = t[10]
    crash = t[100]
    long_bid = path(t, 99.99, [(crash, 20.0)])
    long_ask = long_bid + 0.01
    short_bid = path(t, 101.0, [(crash, 21.0)])
    lg = leg("a", t, long_bid, long_ask, mt=t, mark=(long_bid + long_ask) / 2)
    sh = leg("b", t, short_bid, short_bid + 0.01, mt=t, mark=short_bid)
    r = run_trade(Pair("a>b", "X", "K-PP", lg, sh), sig, Variant(50, 60, "E1_0", 24, tau_bp=-1e6), {**P, "stop_bp": 1e9}, t[-1])
    assert r["reason"] == "liquidation"
    assert r["t_out"] >= crash
    gain_short = (101.0 - 21.01) / 101.0 * 1e4
    assert r["price_bp"] == pytest.approx(-5000 + gain_short)  # the long lost exactly its margin, not 80%


def test_margin_transfer_and_spot_in_full():
    t = t_grid()
    spot = Pair("s>b", "X", "K-SP", leg("a", t, 99, 100, kind="spot"), leg("b", t, 101, 102))
    assert margin_needs(spot, 1000, 2) == {"a": 1000, "b": 500}
    m = Margin(3000, ["a", "b", "c"])  # 1000 each
    assert m.need({"a": 500, "b": 500}) == (0, True)
    assert m.need({"a": 1500, "b": 500}) == (1, True)
    m.commit({"a": 1500, "b": 500})
    assert m.bal == {"a": 0, "b": 500, "c": 500}
    assert m.need({"a": 2000}) == (0, False)


def test_transfer_delays_and_costs():
    t = t_grid()
    p = pair_simple(t=t)
    r = run_trade(p, t[10], V, P, t[-1], transfer_bp=10.0, extra_delay=1800)
    assert r["t_in"] == t[10] + 1800 + 60 + 20
    assert r["transfer_bp"] == 10.0
    assert r["net_bp"] == pytest.approx(r["price_bp"] - r["fees_bp"] + r["funding_bp"] - 10)


def test_median_exit_needs_history():
    t = t_grid()
    e2 = Variant(50, 60, "E2", 24, tau_bp=None)
    assert run_trade(pair_simple(t=t), t[10], e2, P, t[-1])["status"] == "no_history"


def test_holdout_lock(tmp_path, monkeypatch):
    cfgfile = tmp_path / "c.yaml"
    cfgfile.write_text("stageX2:\n  samples: {min_days: 14, dev_days: 7}\n", encoding="utf-8")
    monkeypatch.setattr(common, "CONFIG", cfgfile)
    for name in ("FROZEN", "ATTEMPTS", "HOLDOUT", "SELECTED"):
        monkeypatch.setattr(common, name, tmp_path / name)
    monkeypatch.setattr(common, "LOGS", tmp_path)
    monkeypatch.setattr(common.section_hash, "__defaults__", (cfgfile,))
    monkeypatch.setattr(common.load_config, "__defaults__", (cfgfile,))
    for name in ("HOLDOUT", "SELECTED"):
        monkeypatch.setattr(evaluate, name, tmp_path / name)
    monkeypatch.setattr(evaluate, "DERIVED", tmp_path)
    cfg = common.load_config()
    with pytest.raises(SystemExit):
        evaluate.open_holdout(cfg)  # not frozen
    common.freeze("")
    with pytest.raises(SystemExit):
        evaluate.open_holdout(cfg)  # no chosen variant
    (tmp_path / "SELECTED").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(evaluate.store, "segments", lambda c: (["d"] * 7, ["d"] * 3))
    with pytest.raises(SystemExit):
        evaluate.open_holdout(cfg)  # too short
    with pytest.raises(SystemExit):
        evaluate.holdout(cfg)       # closed
    monkeypatch.setattr(evaluate.store, "segments", lambda c: (["d"] * 7, ["d"] * 7))
    evaluate.open_holdout(cfg)
    with pytest.raises(SystemExit):
        evaluate.open_holdout(cfg)  # only once
    with pytest.raises(SystemExit):
        common.freeze("change after opening")
    cfgfile.write_text("stageX2:\n  samples: {min_days: 1, dev_days: 7}\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        common.require_frozen()     # the rules moved


def test_bootstrap_is_by_day():
    tr = pd.DataFrame({"t_in": [0, 1, 86400, 86401], "net_bp": [10.0, 10.0, -10.0, -10.0]})
    lo, hi = evaluate.block_ci(tr, 0.9, 2000, 1)
    assert lo == pytest.approx(-10) and hi == pytest.approx(10)  # whole days move: two days, one sign each


def test_sink_shards_then_merges(tmp_path):
    s = recorder.Sink(tmp_path)
    s.add("snaps", "okx", ("perp", 100, 5, 7))
    s.flush()
    s.add("snaps", "okx", ("perp", 50, 6, 7))
    s.flush()
    day = recorder.day_of(100)
    assert len(list((tmp_path / "snaps" / day).glob("*.shard"))) == 2
    s.compact("2099-01-01")
    files = list((tmp_path / "snaps" / day).glob("*"))
    assert [f.name for f in files] == ["okx.parquet"]
    assert recorder.pq.read_table(files[0])["t"].to_pylist() == [50, 100]


def test_trim_book_covers_the_size():
    p, q = recorder.trim_book([(100, 1), (99, 1), (98, 1)], 0.5, 90)
    assert p == [100, 99] and q == [0.5, 0.5]


def test_scan_finds_candidate_and_respects_unit_check(tmp_path, monkeypatch):
    monkeypatch.setattr(recorder, "RAW", tmp_path)
    cfg = common.load_config()
    r = recorder.Recorder(cfg)
    a, b, c = ("okx", "perp", "A"), ("bybit", "perp", "B"), ("xt", "spot", "C")
    for k, base, ct in ((a, "ZZZ", 1), (b, "1000ZZZ", 1), (c, "ZZZ", 1)):
        r.insts[k] = recorder.Inst(k[0], k[1], k[2], base, "USDT", ct)
    t = 1_800_000_000
    r.latest[a] = (100.0, 100.1, t, None)
    r.latest[b] = (100510.0, 100520.0, t, None)   # 1000ZZZ: 100.51 per ZZZ, 41 bp above A's ask
    r.latest[c] = (500.0, 500.1, t, None)         # another token under one ticker
    r.scan_once(t)
    assert (a, b) in r.cand
    assert not any(c in k for k in r.cand)
    rows = pd.read_csv(tmp_path / "candidates.csv")
    assert rows["event"].tolist() == ["start"]
    r2 = recorder.Recorder(cfg)
    r2.restore_candidates()
    assert (a, b) in r2.cand
    r.latest[b] = (100.0, 100.1, t + 15, None)
    r.latest[a] = (100.0, 100.1, t + 15, None)
    r.scan_once(t + 15)
    r.scan_once(t + 15 + cfg["recording"]["cand_off_s"])
    assert (a, b) in r.cand  # quotes went stale: nothing was seen below the threshold for 30 minutes
    r.latest[a] = r.latest[b] = (100.0, 100.1, t + 2000, None)
    r.scan_once(t + 2000)
    r.latest[a] = r.latest[b] = (100.0, 100.1, t + 2000 + 1800, None)
    r.scan_once(t + 2000 + 1800)
    assert (a, b) not in r.cand


def test_config_is_complete():
    cfg = common.load_config()
    assert set(cfg["fees_bp"]) == set(recorder.VENUES) == set(cfg["recording"]["book_rps"])
    assert len(paper.variants(cfg["paper"])) == 54
    assert cfg["recording"]["cand_on_bp"] == 30
    assert json.dumps(cfg)
