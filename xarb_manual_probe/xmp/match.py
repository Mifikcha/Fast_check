"""One asset across venues: ticker normalization, contract unit check, pair categories. Pure functions."""
import re
import statistics

KILO = {"KPEPE", "KSHIB", "KBONK", "KLUNC", "KFLOKI", "KDOGS", "KNEIRO"}


def parse_base(base, venue=""):
    """Same rules as the screener (BFS `parseBase`): 1000PEPE, SHIB1000, 1MBABYDOGE, kPEPE quote many tokens per unit."""
    b = base.upper()
    if venue in ("hyperliquid", "paradex") and b in KILO:
        return b[1:], 1000
    m = re.fullmatch(r"(10{2,9})([A-Z][A-Z0-9]+)", b)
    if m:
        return m[2], int(m[1])
    m = re.fullmatch(r"([A-Z]{2,})(10{3,9})", b)  # from 1000: US100 is an index, not a hundred of US
    if m:
        return m[1], int(m[2])
    m = re.fullmatch(r"1M([A-Z]{3,})", b)
    if m:
        return m[1], 1_000_000
    return b, 1


def unit_price(price, mult, quote, usdc_usdt):
    """USDT per one unit of the asset. USDC legs convert at the rate of the same moment."""
    return price / mult * (usdc_usdt if quote == "USDC" else 1.0)


def pair_ok(gaps, limit=0.02):
    """Contract unit and identity check: the median relative price gap of the pair stays within 2% (as in F2). Past samples only."""
    return bool(gaps) and abs(statistics.median(gaps)) <= limit


def can_pair(long_kind, short_kind):
    """A spot leg can only be the long one: shorting spot takes a margin account."""
    return short_kind == "perp"


def category(asset, long_venue, long_kind, short_venue, short_kind, cat):
    """Frozen priority: K-SYN > K-DEX > K-SP > K-PP."""
    syn = set(cat["syn_assets"])
    if asset in syn or (asset.endswith("X") and asset[:-1] in syn) or re.fullmatch(cat["syn_regex"], asset):
        return "K-SYN"
    if long_venue in cat["dex"] or short_venue in cat["dex"]:
        return "K-DEX"
    return "K-SP" if long_kind == "spot" else "K-PP"


def flags(vol_long, vol_short, listed_ms, now_ms, wallet_open, cat):
    out = []
    if min(vol_long or 0, vol_short or 0) < cat["low_turnover_usd"]:
        out.append("low_turnover")
    if wallet_open is not True:
        out.append("wallet_closed_or_unknown")
    if listed_ms and now_ms - listed_ms < cat["new_listing_days"] * 86_400_000:
        out.append("new_listing")
    return out
