"""Public, keyless market data of the screener's venues. No orders, no accounts."""
import base64
import gzip
import http.client
import json
import os
import re
import socket
import ssl
import struct
import threading
import time
import urllib.parse
from dataclasses import dataclass
from typing import Callable

UA = {"User-Agent": "Mozilla/5.0", "Content-Type": "application/json", "Accept-Encoding": "gzip"}


_local = threading.local()


class RateLimited(Exception):
    def __init__(self, retry_after):
        super().__init__(f"429, retry after {retry_after}s")
        self.retry_after = retry_after


def get(url, body=None, timeout=20):
    """Keep-alive per thread: a fresh TLS handshake per call costs more than the rate limits do. Symbols may be non-ASCII."""
    u = urllib.parse.urlsplit(urllib.parse.quote(url, safe=":/?&=%+,~"))
    conns = _local.__dict__.setdefault("conns", {})
    path = f"{u.path}?{u.query}" if u.query else u.path
    data = json.dumps(body).encode() if body is not None else None
    for attempt in (0, 1):
        c = conns.get(u.netloc) or conns.setdefault(u.netloc, http.client.HTTPSConnection(u.netloc, timeout=timeout))
        try:
            c.request("POST" if data else "GET", path, data, UA)
            r = c.getresponse()
            raw = r.read()
            break
        except (http.client.HTTPException, OSError):  # the venue dropped an idle connection
            c.close()
            conns.pop(u.netloc, None)
            if attempt:
                raise
    if r.status in (418, 429):
        raise RateLimited(float(r.getheader("Retry-After") or 60))
    if r.status != 200:
        raise ConnectionError(f"{u.netloc}: HTTP {r.status}")
    return json.loads(gzip.decompress(raw) if r.getheader("Content-Encoding") == "gzip" else raw)


def num(v):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if x == x and abs(x) != float("inf") else None


def pos(v):
    x = num(v)
    return x if x and x > 0 else None


@dataclass
class Inst:
    venue: str
    kind: str          # spot | perp
    native: str
    base: str          # as the venue writes it, 1000PEPE stays 1000PEPE
    quote: str         # USDT | USDC
    ct: float = 1.0    # base units per contract: book and BBO sizes are multiplied by it
    listed_ms: int | None = None
    asset_class: str = ""


def okx_insts():
    out = []
    for t in ("SWAP", "SPOT"):
        for i in get(f"https://www.okx.com/api/v5/public/instruments?instType={t}")["data"]:
            if i["state"] != "live":
                continue
            if t == "SWAP":
                if i["ctType"] == "linear" and i["settleCcy"] in ("USDT", "USDC"):
                    out.append(Inst("okx", "perp", i["instId"], i["ctValCcy"], i["settleCcy"], float(i["ctVal"]) * float(i.get("ctMult") or 1), int(i["listTime"] or 0) or None))
            elif i["quoteCcy"] in ("USDT", "USDC"):
                out.append(Inst("okx", "spot", i["instId"], i["baseCcy"], i["quoteCcy"], 1.0, int(i["listTime"] or 0) or None))
    return out


def okx_tickers(kind):
    t = "SWAP" if kind == "perp" else "SPOT"
    return [(r["instId"], pos(r["bidPx"]), pos(r["askPx"]), pos(r["bidSz"]), pos(r["askSz"]), int(r["ts"]))
            for r in get(f"https://www.okx.com/api/v5/market/tickers?instType={t}")["data"]]


def okx_stats():
    s: dict[str, dict] = {}
    for r in get("https://www.okx.com/api/v5/public/funding-rate?instId=ANY")["data"]:
        now, nxt = int(r["fundingTime"]), int(r.get("nextFundingTime") or 0)
        s[r["instId"]] = {"funding": num(r["fundingRate"]), "next_ms": now, "interval_h": (nxt - now) / 3.6e6 if nxt else None}
    for r in get("https://www.okx.com/api/v5/public/mark-price?instType=SWAP")["data"]:
        s.setdefault(r["instId"], {})["mark"] = num(r["markPx"])
    for r in get("https://www.okx.com/api/v5/public/open-interest?instType=SWAP")["data"]:
        s.setdefault(r["instId"], {})["oi"] = num(r["oiUsd"])
    for r in get("https://www.okx.com/api/v5/market/tickers?instType=SWAP")["data"]:
        s.setdefault(r["instId"], {})["vol"] = (num(r["volCcy24h"]) or 0) * (num(r["last"]) or 0)
    for q in ("USDT", "USDC"):
        idx = {r["instId"]: num(r["idxPx"]) for r in get(f"https://www.okx.com/api/v5/market/index-tickers?quoteCcy={q}")["data"]}
        for k, v in s.items():
            if k.endswith(f"-{q}-SWAP"):
                v["index"] = idx.get(k[:-5])
    return s


def bybit_insts():
    out = []
    for cat, kind in (("linear", "perp"), ("spot", "spot")):
        cursor = ""
        while True:
            r = get(f"https://api.bybit.com/v5/market/instruments-info?category={cat}&limit=1000&cursor={urllib.parse.quote(cursor)}")["result"]
            for i in r["list"]:
                if i["status"] == "Trading" and i["quoteCoin"] in ("USDT", "USDC") and (kind == "spot" or i.get("contractType") == "LinearPerpetual"):
                    out.append(Inst("bybit", kind, i["symbol"], i["baseCoin"], i["quoteCoin"], 1.0, int(i.get("launchTime") or 0) or None))
            cursor = r.get("nextPageCursor") or ""
            if not cursor:
                break
    return out


def bybit_tickers(kind):
    c = "linear" if kind == "perp" else "spot"
    return [(r["symbol"], pos(r["bid1Price"]), pos(r["ask1Price"]), pos(r["bid1Size"]), pos(r["ask1Size"]), None)
            for r in get(f"https://api.bybit.com/v5/market/tickers?category={c}")["result"]["list"]]


def bybit_stats():
    return {r["symbol"]: {"funding": num(r.get("fundingRate")), "next_ms": num(r.get("nextFundingTime")), "interval_h": num(r.get("fundingIntervalHour")),
                          "mark": num(r.get("markPrice")), "index": num(r.get("indexPrice")), "vol": num(r.get("turnover24h")), "oi": num(r.get("openInterestValue"))}
            for r in get("https://api.bybit.com/v5/market/tickers?category=linear")["result"]["list"]}


def bingx_insts():
    out = [Inst("bingx", "perp", i["symbol"], i["asset"], i["currency"], 1.0, num(i.get("launchTime")))
           for i in get("https://open-api.bingx.com/openApi/swap/v2/quote/contracts")["data"]
           if i.get("status") == 1 and i.get("currency") in ("USDT", "USDC")]
    for r in get(f"https://open-api.bingx.com/openApi/spot/v1/ticker/24hr?timestamp={int(time.time() * 1000)}")["data"]:
        b, q = r["symbol"].split("-", 1)
        if q in ("USDT", "USDC"):
            out.append(Inst("bingx", "spot", r["symbol"], b, q))
    return out


def bingx_tickers(kind):
    u = "swap/v2/quote/ticker?" if kind == "perp" else "spot/v1/ticker/24hr?"
    return [(r["symbol"], pos(r.get("bidPrice")), pos(r.get("askPrice")), pos(r.get("bidQty")), pos(r.get("askQty")), r.get("closeTime"))
            for r in get(f"https://open-api.bingx.com/openApi/{u}timestamp={int(time.time() * 1000)}")["data"]]


def bingx_stats():
    s = {r["symbol"]: {"funding": num(r["lastFundingRate"]), "next_ms": num(r["nextFundingTime"]), "interval_h": num(r.get("fundingIntervalHours")),
                       "mark": num(r["markPrice"]), "index": num(r["indexPrice"])}
         for r in get("https://open-api.bingx.com/openApi/swap/v2/quote/premiumIndex")["data"]}
    for r in get("https://open-api.bingx.com/openApi/swap/v2/quote/ticker")["data"]:
        s.setdefault(r["symbol"], {})["vol"] = num(r["quoteVolume"])
    return s


def aster_insts():
    out = []
    for host, path, kind in (("fapi", "fapi/v1", "perp"), ("sapi", "api/v1", "spot")):
        for i in get(f"https://{host}.asterdex.com/{path}/exchangeInfo")["symbols"]:
            if i["status"] == "TRADING" and i["quoteAsset"] in ("USDT", "USDC") and (kind == "spot" or i.get("contractType") == "PERPETUAL"):
                out.append(Inst("aster", kind, i["symbol"], i["baseAsset"], i["quoteAsset"], 1.0, i.get("onboardDate")))
    return out


def aster_tickers(kind):
    u = "fapi.asterdex.com/fapi/v1" if kind == "perp" else "sapi.asterdex.com/api/v1"
    return [(r["symbol"], pos(r["bidPrice"]), pos(r["askPrice"]), pos(r["bidQty"]), pos(r["askQty"]), r.get("time"))
            for r in get(f"https://{u}/ticker/bookTicker")]


def aster_stats():
    u = "https://fapi.asterdex.com/fapi/v1"
    s = {r["symbol"]: {"funding": num(r["lastFundingRate"]), "next_ms": num(r["nextFundingTime"]), "mark": num(r["markPrice"]), "index": num(r["indexPrice"])}
         for r in get(f"{u}/premiumIndex")}
    for r in get(f"{u}/fundingInfo"):
        s.setdefault(r["symbol"], {})["interval_h"] = num(r["fundingIntervalHours"])
    for r in get(f"{u}/ticker/24hr"):
        s.setdefault(r["symbol"], {})["vol"] = num(r["quoteVolume"])
    return s


HL = "https://api.hyperliquid.xyz/info"


def hl_rows():
    meta, ctx = get(HL, {"type": "metaAndAssetCtxs"})
    return [(u, c) for u, c in zip(meta["universe"], ctx) if not u.get("isDelisted")]


def hl_insts():
    return [Inst("hyperliquid", "perp", u["name"], u["name"], "USDC") for u, _ in hl_rows()]


def hl_tickers(kind):
    # shortcut: impact prices stand in for BBO (no bulk BBO, spot gives a mid only), a per-coin l2Book poll would break the rate limit.
    return [(u["name"], pos(c["impactPxs"][0]), pos(c["impactPxs"][1]), None, None, None) for u, c in hl_rows() if c.get("impactPxs")]


def hl_stats():
    nxt = (int(time.time()) // 3600 + 1) * 3_600_000
    return {u["name"]: {"funding": num(c["funding"]), "next_ms": nxt, "interval_h": 1.0, "mark": num(c["markPx"]), "index": num(c["oraclePx"]),
                        "vol": num(c["dayNtlVlm"]), "oi": (num(c["openInterest"]) or 0) * (num(c["markPx"]) or 0)} for u, c in hl_rows()}


def wb_insts():
    out = [Inst("whitebit", "perp", r["ticker_id"], r["stock_currency"], r["money_currency"])
           for r in get("https://whitebit.com/api/v4/public/futures")["result"] if r["money_currency"] in ("USDT", "USDC")]
    for r in get("https://whitebit.com/api/v2/public/ticker")["result"]:
        b, q = r["tradingPairs"].rsplit("_", 1)
        if q in ("USDT", "USDC") and r.get("tradesEnabled"):
            out.append(Inst("whitebit", "spot", r["tradingPairs"], b, q))
    return out


def wb_tickers(kind):
    if kind == "perp":
        return [(r["ticker_id"], pos(r["bid"]), pos(r["ask"]), None, None, None) for r in get("https://whitebit.com/api/v4/public/futures")["result"]]
    return [(r["tradingPairs"], pos(r["highestBid"]), pos(r["lowestAsk"]), None, None, None) for r in get("https://whitebit.com/api/v2/public/ticker")["result"]]


def wb_stats():
    return {r["ticker_id"]: {"funding": num(r["funding_rate"]), "next_ms": num(r["next_funding_rate_timestamp"]), "interval_h": (num(r.get("funding_interval_minutes")) or 480) / 60,
                             "mark": num(r["mark_price"]), "index": num(r["index_price"]), "vol": num(r["money_volume"]),
                             "oi": (num(r["open_interest"]) or 0) * (num(r["mark_price"]) or 0)}
            for r in get("https://whitebit.com/api/v4/public/futures")["result"]}


def bfx_rows():
    return [r for r in get("https://api-pub.bitfinex.com/v2/tickers?symbols=ALL") if r[0].startswith("t")]


def bfx_parse(sym):
    """tBTCF0:USTF0 -> perp BTC; tBTCUST and tDOGE:UST -> spot; the rest (real USD, BTC pairs) is out."""
    s = sym[1:]
    if s.endswith("F0:USTF0"):
        return "perp", s[:-8]
    if s.endswith("UST"):
        return "spot", s[:-3].rstrip(":")
    return None, None


def bfx_insts():
    out = []
    for r in bfx_rows():
        kind, base = bfx_parse(r[0])
        if kind:
            out.append(Inst("bitfinex", kind, r[0], base, "USDT"))
    return out


def bfx_tickers(kind):
    return [(r[0], pos(r[1]), pos(r[3]), pos(r[2]), pos(r[4]), None) for r in bfx_rows() if bfx_parse(r[0])[0] == kind]


def bfx_stats():
    s = {r[0]: {"funding": num(r[12]), "next_ms": num(r[8]), "interval_h": 8.0, "mark": num(r[15]), "index": num(r[4]), "oi": (num(r[18]) or 0) * (num(r[15]) or 0)}
         for r in get("https://api-pub.bitfinex.com/v2/status/deriv?keys=ALL")}
    for r in bfx_rows():
        if r[0] in s:
            s[r[0]]["vol"] = (num(r[8]) or 0) * (num(r[7]) or 0)
    return s


BLOFIN = "https://openapi.blofin.com/api/v1/market"


def blofin_insts():
    return [Inst("blofin", "perp", i["instId"], i["baseCurrency"], i["quoteCurrency"], float(i["contractValue"]), int(i["listTime"]), i.get("assetClass", ""))
            for i in get(f"{BLOFIN}/instruments")["data"] if i["state"] == "live" and i["contractType"] == "linear"]


def blofin_tickers(kind):
    return [(r["instId"], pos(r["bidPrice"]), pos(r["askPrice"]), pos(r["bidSize"]), pos(r["askSize"]), int(r["ts"])) for r in get(f"{BLOFIN}/tickers")["data"]]


def blofin_stats():
    s = {r["instId"]: {"funding": num(r["fundingRate"]), "next_ms": num(r["fundingTime"]),
                       "interval_h": (num(r["fundingInterval"]) or 8) / (60 if r["fundingIntervalUnit"].startswith("min") else 1)}
         for r in get(f"{BLOFIN}/funding-rate")["data"]}
    for r in get(f"{BLOFIN}/mark-price")["data"]:
        s.setdefault(r["instId"], {}).update(mark=num(r["markPrice"]), index=num(r["indexPrice"]))
    for r in get(f"{BLOFIN}/tickers")["data"]:
        s.setdefault(r["instId"], {})["vol"] = (num(r["volCurrency24h"]) or 0) * (num(r["last"]) or 0)
    return s


XTF = "https://fapi.xt.com/future/market/v1/public"


def xt_contracts():
    r = get(f"{XTF}/cg/contracts")
    return r["data"] if isinstance(r, dict) else r


def xt_insts():
    out = [Inst("xt", "perp", r["symbol"], r["base_currency"].upper(), r["target_currency"].upper(), float(r["contractSize"]), r.get("start_timestamp"))
           for r in xt_contracts() if r["target_currency"].upper() in ("USDT", "USDC") and not re.search(r"_\d{6}$", r["symbol"])]  # eth_usdt_270326 is dated
    for r in get("https://sapi.xt.com/v4/public/ticker/book")["result"]:
        b, q = r["s"].upper().rsplit("_", 1)
        if q in ("USDT", "USDC"):
            out.append(Inst("xt", "spot", r["s"], b, q))
    return out


def xt_tickers(kind):
    if kind == "perp":
        return [(r["s"], pos(r.get("bp")), pos(r.get("ap")), None, None, r.get("t")) for r in get(f"{XTF}/q/agg-tickers")["result"]]
    return [(r["s"], pos(r["bp"]), pos(r["ap"]), pos(r["bq"]), pos(r["aq"]), r.get("t")) for r in get("https://sapi.xt.com/v4/public/ticker/book")["result"]]


def xt_stats():
    s = {r["symbol"]: {"funding": num(r["funding_rate"]), "next_ms": num(r["next_funding_rate_timestamp"]), "interval_h": (num(r.get("funding_interval_minutes")) or 480) / 60,
                       "index": num(r["index_price"]), "vol": num(r["target_volume"]), "oi": num(r["open_interest_usd"])} for r in xt_contracts()}
    for r in get(f"{XTF}/q/agg-tickers")["result"]:
        if r["s"] in s:
            s[r["s"]]["mark"] = num(r.get("m"))
    return s


def paradex_rows():
    return [r for r in get("https://api.prod.paradex.trade/v1/markets/summary?market=ALL")["results"] if r["symbol"].endswith("-PERP")]


def paradex_insts():
    return [Inst("paradex", "perp", r["symbol"], r["symbol"].split("-")[0], "USDC", 1.0, r.get("created_at")) for r in paradex_rows()]


def paradex_tickers(kind):
    return [(r["symbol"], pos(r["bid"]), pos(r["ask"]), pos(r["bid_size"]), pos(r["ask_size"]), None) for r in paradex_rows()]


def paradex_stats():
    return {r["symbol"]: {"funding": num(r["funding_rate"]), "interval_h": (num(r.get("funding_interval_minutes")) or 480) / 60, "mark": num(r["mark_price"]),
                          "index": num(r["underlying_price"]), "vol": num(r["volume_24h"]), "oi": (num(r["open_interest"]) or 0) * (num(r["mark_price"]) or 0)} for r in paradex_rows()}


LIGHTER = "https://mainnet.zklighter.elliot.ai/api/v1"


def lighter_rows():
    return [m for m in get(f"{LIGHTER}/orderBookDetails")["order_book_details"] if m["status"] == "active"]


def lighter_insts():
    return [Inst("lighter", m["market_type"], str(m["market_id"]), m["symbol"].split("/")[0], "USDC") for m in lighter_rows()
            if m["market_type"] == "perp" or m["symbol"].endswith("/USDC")]


def lighter_stats():
    nxt = (int(time.time()) // 3600 + 1) * 3_600_000
    s = {str(m["market_id"]): {"mark": num(m.get("mark_price")), "index": num(m.get("index_price")), "vol": num(m.get("daily_quote_token_volume")), "oi": num(m.get("open_interest_usd")),
                               "interval_h": 1.0, "next_ms": nxt} for m in lighter_rows()}
    for f in get(f"{LIGHTER}/funding-rates")["funding_rates"]:
        if f["exchange"] == "lighter" and str(f["market_id"]) in s:
            s[str(f["market_id"])]["funding"] = num(f["rate"]) / 8  # the list normalizes every rate to 8 h, Lighter settles hourly
    return s


BITUNIX = "https://fapi.bitunix.com/api/v1/futures/market"


def bitunix_insts():
    return [Inst("bitunix", "perp", p["symbol"], p["base"], p["quote"]) for p in get(f"{BITUNIX}/trading_pairs")["data"]
            if p["symbolStatus"] == "OPEN" and p["quote"] in ("USDT", "USDC")]


def bitunix_stats():
    s = {r["symbol"]: {"funding": (num(r["fundingRate"]) or 0) / 100, "next_ms": num(r["nextFundingTime"]), "interval_h": num(r["fundingInterval"]),
                       "mark": num(r["markPrice"]), "index": num(r["indexPrice"])} for r in get(f"{BITUNIX}/funding_rate/batch")["data"]}  # reports percent
    for r in get(f"{BITUNIX}/tickers")["data"]:
        s.setdefault(r["symbol"], {})["vol"] = num(r["quoteVol"])
    return s


PIONEX = "https://api.pionex.com/api/v1"
_pionex_funding: dict[str, dict] = {}
_pionex_at = 0


def pionex_insts():
    out = []
    for t, kind in (("SPOT", "spot"), ("PERP", "perp")):
        for s in get(f"{PIONEX}/common/symbols?type={t}")["data"]["symbols"]:
            base, quote = s["symbol"].split("_")[:2]
            if quote in ("USDT", "USDC") and s.get("enable", True):
                out.append(Inst("pionex", kind, s["symbol"], base, quote))
    return out


def pionex_tickers(kind):
    return [(r["symbol"], pos(r["bidPrice"]), pos(r["askPrice"]), pos(r["bidSize"]), pos(r["askSize"]), r.get("timestamp"))
            for r in get(f"{PIONEX}/market/bookTickers?type={kind.upper()}")["data"]["tickers"]]


def pionex_stats():
    """Funding is per symbol only: a slice each minute, so the whole list turns over in about 25 minutes. No bulk mark: the sim falls back to mid."""
    global _pionex_at
    vols = {r["symbol"]: num(r["amount"]) for r in get(f"{PIONEX}/market/tickers?type=PERP")["data"]["tickers"]}
    names = sorted(vols)
    for sym in names[_pionex_at:_pionex_at + 25]:
        rates = get(f"{PIONEX}/market/fundingRates?symbol={sym}&limit=2")["data"]["rates"]
        if rates:
            iv = (rates[0]["fundingTime"] - rates[1]["fundingTime"]) / 3.6e6 if len(rates) > 1 else None
            _pionex_funding[sym] = {"funding": num(rates[0]["fundingRate"]), "interval_h": iv, "next_ms": rates[0]["fundingTime"] + iv * 3.6e6 if iv else None}
        time.sleep(0.1)
    _pionex_at = 0 if _pionex_at + 25 >= len(names) else _pionex_at + 25
    return {k: {"vol": v, **_pionex_funding.get(k, {})} for k, v in vols.items()}


def _pairs(rows):
    return [(float(r[0]), float(r[1])) for r in rows]


def book_okx(kind, n):
    d = get(f"https://www.okx.com/api/v5/market/books?instId={n}&sz=400")["data"][0]
    return _pairs(d["bids"]), _pairs(d["asks"])


def book_bybit(kind, n):
    d = get(f"https://api.bybit.com/v5/market/orderbook?category={'linear' if kind == 'perp' else 'spot'}&symbol={n}&limit=200")["result"]
    return _pairs(d["b"]), _pairs(d["a"])


def book_bingx(kind, n):
    u = "swap/v2/quote/depth" if kind == "perp" else "spot/v1/market/depth"
    d = get(f"https://open-api.bingx.com/openApi/{u}?symbol={n}&limit=100&timestamp={int(time.time() * 1000)}")["data"]
    return _pairs(d["bids"]), sorted(_pairs(d["asks"]))


def book_aster(kind, n):
    u = "fapi.asterdex.com/fapi/v1" if kind == "perp" else "sapi.asterdex.com/api/v1"
    d = get(f"https://{u}/depth?symbol={n}&limit=100")
    return _pairs(d["bids"]), _pairs(d["asks"])


def book_hl(kind, n):
    b, a = get(HL, {"type": "l2Book", "coin": n})["levels"]
    return [(float(x["px"]), float(x["sz"])) for x in b], [(float(x["px"]), float(x["sz"])) for x in a]


def book_wb(kind, n):
    d = get(f"https://whitebit.com/api/v4/public/orderbook/{n}?limit=100")
    return _pairs(d["bids"]), _pairs(d["asks"])


def book_bfx(kind, n):
    rows = get(f"https://api-pub.bitfinex.com/v2/book/{urllib.parse.quote(n)}/P0?len=100")
    return [(float(p), a) for p, _, a in rows if a > 0], [(float(p), -a) for p, _, a in rows if a < 0]


def book_blofin(kind, n):
    d = get(f"{BLOFIN}/books?instId={n}&size=100")["data"][0]
    return _pairs(d["bids"]), _pairs(d["asks"])


def book_xt(kind, n):
    if kind == "perp":
        d = get(f"{XTF}/q/depth?symbol={n}&level=50")["result"]
        return _pairs(d["b"]), _pairs(d["a"])
    d = get(f"https://sapi.xt.com/v4/public/depth?symbol={n}&limit=100")["result"]
    return _pairs(d["bids"]), _pairs(d["asks"])


def book_paradex(kind, n):
    d = get(f"https://api.prod.paradex.trade/v1/orderbook/{n}?depth=100")
    return _pairs(d["bids"]), _pairs(d["asks"])


def book_lighter(kind, n):
    d = get(f"{LIGHTER}/orderBookOrders?market_id={n}&limit=250")
    side = lambda rows: [(float(o["price"]), float(o["remaining_base_amount"])) for o in rows]
    return sorted(side(d["bids"]), reverse=True), sorted(side(d["asks"]))


def book_bitunix(kind, n):
    d = get(f"{BITUNIX}/depth?symbol={n}&limit=max")["data"]
    return sorted(_pairs(d["bids"]), reverse=True), sorted(_pairs(d["asks"]))


def book_pionex(kind, n):
    d = get(f"{PIONEX}/market/depth?symbol={n}&limit=100")["data"]
    return _pairs(d["bids"]), _pairs(d["asks"])


def _ms(rows, t, r):
    return [(int(x[t]), float(x[r])) for x in rows]


def hist_okx(n):
    return _ms(get(f"https://www.okx.com/api/v5/public/funding-rate-history?instId={n}&limit=100")["data"], "fundingTime", "realizedRate")


def hist_bybit(n):
    return _ms(get(f"https://api.bybit.com/v5/market/funding/history?category=linear&symbol={n}&limit=200")["result"]["list"], "fundingRateTimestamp", "fundingRate")


def hist_bingx(n):
    return _ms(get(f"https://open-api.bingx.com/openApi/swap/v2/quote/fundingRate?symbol={n}&limit=100")["data"], "fundingTime", "fundingRate")


def hist_aster(n):
    return _ms(get(f"https://fapi.asterdex.com/fapi/v1/fundingRate?symbol={n}&limit=100"), "fundingTime", "fundingRate")


def hist_hl(n):
    return _ms(get(HL, {"type": "fundingHistory", "coin": n, "startTime": int(time.time() * 1000) - 96 * 3_600_000}), "time", "fundingRate")


def hist_wb(n):
    return [(int(x["fundingTime"]) * 1000, float(x["fundingRate"])) for x in get(f"https://whitebit.com/api/v4/public/funding-history/{n}?limit=100")]


def hist_blofin(n):
    return _ms(get(f"{BLOFIN}/funding-rate-history?instId={n}&limit=100")["data"], "fundingTime", "fundingRate")


def hist_xt(n):
    return _ms(get(f"{XTF}/q/funding-rate-record?symbol={n}&limit=100")["result"]["items"], "createdTime", "fundingRate")


def hist_lighter(n):
    now = int(time.time())
    rows = get(f"{LIGHTER}/fundings?market_id={n}&resolution=1h&start_timestamp={now - 96 * 3600}&end_timestamp={now}&count_back=96")["fundings"]
    return [(int(x["timestamp"]) * 1000, float(x["rate"]) / 100 * (1 if x["direction"] == "long" else -1)) for x in rows]  # percent, "long" = longs pay


def hist_pionex(n):
    return _ms(get(f"{PIONEX}/market/fundingRates?symbol={n}&limit=100")["data"]["rates"], "fundingTime", "fundingRate")


def wallet_wb():
    return [(k, v.get("can_deposit"), v.get("can_withdraw")) for k, v in get("https://whitebit.com/api/v4/public/assets").items()]


def wallet_xt():
    out = []
    for c in get("https://sapi.xt.com/v4/public/wallet/support/currency")["result"]:
        ch = c.get("supportChains") or []
        out.append((c["currency"].upper(), any(x.get("depositEnabled") for x in ch), any(x.get("withdrawEnabled") for x in ch)))
    return out


def ws_frames(url, open_msgs: Callable, on_text: Callable, stop: Callable, idle_s=30):
    """Blocking RFC 6455 client on the standard library: the repo's venv has no WS package. Returns or raises when the socket dies."""
    u = urllib.parse.urlparse(url)
    raw = socket.create_connection((u.hostname, 443), timeout=15)
    s = ssl.create_default_context().wrap_socket(raw, server_hostname=u.hostname)
    key = base64.b64encode(os.urandom(16)).decode()
    path = u.path + (f"?{u.query}" if u.query else "")
    s.sendall(f"GET {path} HTTP/1.1\r\nHost: {u.hostname}\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\nUser-Agent: Mozilla/5.0\r\n\r\n".encode())
    head = b""
    while b"\r\n\r\n" not in head:
        c = s.recv(1)
        if not c:
            raise ConnectionError("no handshake")
        head += c
    if b" 101 " not in head.split(b"\r\n")[0]:
        raise ConnectionError(head[:80])

    def frame(op, data=b""):
        mask = os.urandom(4)
        n = len(data)
        size = bytes([0x80 | n]) if n < 126 else bytes([0x80 | 126]) + struct.pack(">H", n) if n < 65536 else bytes([0x80 | 127]) + struct.pack(">Q", n)
        s.sendall(bytes([0x80 | op]) + size + mask + bytes(b ^ mask[i % 4] for i, b in enumerate(data)))

    def read(n):
        out = b""
        while len(out) < n:
            c = s.recv(n - len(out))
            if not c:
                raise ConnectionError("closed")
            out += c
        return out

    send = lambda m: frame(1, json.dumps(m).encode())
    open_msgs(send)
    s.settimeout(idle_s)
    buf = b""
    try:
        while not stop():
            b0, b1 = read(2)
            n = b1 & 0x7F
            if n == 126:
                n = struct.unpack(">H", read(2))[0]
            elif n == 127:
                n = struct.unpack(">Q", read(8))[0]
            data = read(n)
            op = b0 & 0x0F
            if op == 8:
                raise ConnectionError("close frame")
            if op == 9:
                frame(10, data)
            elif op in (0, 1, 2):
                buf += data
                if b0 & 0x80:
                    on_text(buf.decode(), send)
                    buf = b""
    finally:
        s.close()


def stream_lighter(insts, put, stop):
    ids = [i.native for i in insts]
    last = [time.time()]

    def text(t, send):
        m = json.loads(t)
        if time.time() - last[0] > 30:  # a silent client is dropped after 120 s
            last[0] = time.time()
            send({"type": "ping"})
        if m.get("type") == "ping":
            return send({"type": "pong"})
        ch, tk = m.get("channel", ""), m.get("ticker")
        if ch.startswith("ticker:") and tk:
            put(ch[7:], pos(tk.get("b", {}).get("price")), pos(tk.get("a", {}).get("price")), pos(tk.get("b", {}).get("size")), pos(tk.get("a", {}).get("size")),
                tk["last_updated_at"] // 1000 if tk.get("last_updated_at") else None)

    ws_frames("wss://mainnet.zklighter.elliot.ai/stream?readonly=true", lambda send: [send({"type": "subscribe", "channel": f"ticker/{i}"}) for i in ids], text, stop)


def stream_bitunix(insts, put, stop):
    ids = [i.native for i in insts]
    last = [time.time()]

    def opened(send):
        for k in range(0, len(ids), 300):
            send({"op": "subscribe", "args": [{"symbol": s, "ch": "tickers"} for s in ids[k:k + 300]]})

    def text(t, send):
        m = json.loads(t)
        if time.time() - last[0] > 15:
            last[0] = time.time()
            send({"op": "ping", "ping": int(last[0])})
        if m.get("ch") == "tickers":
            for r in m.get("data") or []:
                put(r["s"], pos(r.get("bd")), pos(r.get("ak")), pos(r.get("bv")), pos(r.get("av")), m.get("ts"))

    ws_frames("wss://fapi.bitunix.com/public/", opened, text, stop)


@dataclass
class Venue:
    insts: Callable
    stats: Callable
    book: Callable
    tickers: Callable | None = None    # None: BBO arrives on `stream`
    stream: Callable | None = None
    hist: Callable | None = None       # None: settled funding comes from the 5-minute snapshots
    socket_size: int = 10_000          # instruments on one stream socket
    wallet: Callable | None = None
    kinds: tuple = ("perp", "spot")


VENUES = {
    "okx": Venue(okx_insts, okx_stats, book_okx, okx_tickers, hist=hist_okx),
    "bybit": Venue(bybit_insts, bybit_stats, book_bybit, bybit_tickers, hist=hist_bybit),
    "bingx": Venue(bingx_insts, bingx_stats, book_bingx, bingx_tickers, hist=hist_bingx),
    "hyperliquid": Venue(hl_insts, hl_stats, book_hl, hl_tickers, hist=hist_hl, kinds=("perp",)),
    "aster": Venue(aster_insts, aster_stats, book_aster, aster_tickers, hist=hist_aster),
    "whitebit": Venue(wb_insts, wb_stats, book_wb, wb_tickers, hist=hist_wb, wallet=wallet_wb),
    "bitfinex": Venue(bfx_insts, bfx_stats, book_bfx, bfx_tickers),
    "blofin": Venue(blofin_insts, blofin_stats, book_blofin, blofin_tickers, hist=hist_blofin, kinds=("perp",)),
    "xt": Venue(xt_insts, xt_stats, book_xt, xt_tickers, hist=hist_xt, wallet=wallet_xt),
    "paradex": Venue(paradex_insts, paradex_stats, book_paradex, paradex_tickers, kinds=("perp",)),
    "lighter": Venue(lighter_insts, lighter_stats, book_lighter, stream=stream_lighter, hist=hist_lighter),
    "bitunix": Venue(bitunix_insts, bitunix_stats, book_bitunix, stream=stream_bitunix, socket_size=300, kinds=("perp",)),
    "pionex": Venue(pionex_insts, pionex_stats, book_pionex, pionex_tickers, hist=hist_pionex),
}
