"""Mark-price data for F2b: hourly mark klines (raw, write once) -> 4-hour mark OHLC panels on the F2 bar grid."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import numpy as np
import pandas as pd

from .common import CLEAN, RAW, Rest, now_utc, periods
from .download import _call, _write_once, read_raw, universe, windows

MARK_PATH = "/v5/market/mark-price-kline"


def mark_kline(rest: Rest, sym: str, start_ms: int, end_ms: int, limit: int) -> None:
    path = RAW / "kline60" / "mark" / f"{sym}.jsonl.gz"
    if path.exists():
        return
    recs = [_call(rest, MARK_PATH, {"category": "linear", "symbol": sym, "interval": 60, "start": lo, "end": hi, "limit": limit})[0]
            for lo, hi in windows(start_ms, end_ms, limit)]
    _write_once(path, recs)


def download_mark(cfg: dict[str, Any], log=print) -> None:
    rest = Rest(cfg["api"])
    p = periods(cfg)
    s_ms, e_ms = int(p["data_start"].value // 10**6), int(p["end"].value // 10**6)
    syms = list(universe(cfg)["symbol"])
    failed = 0
    with ThreadPoolExecutor(int(cfg["api"]["threads"])) as pool:
        futs = {pool.submit(mark_kline, rest, s, s_ms, e_ms, cfg["api"]["kline_limit"]): s for s in syms}
        for f in as_completed(futs):
            try:
                f.result()
            except Exception as exc:  # a re-run fills what is missing
                failed += 1
                log(f"FAILED {futs[f]}: {exc}")
    log(f"{now_utc()} mark download finished: {len(syms)} symbols, failed={failed}, requests={rest.requests}")


def hourly_mark(syms: list[str]) -> pd.DataFrame:
    """Long table symbol, ts, open, high, low, close of hourly mark candles (duplicates dropped)."""
    out = []
    for s in syms:
        path = RAW / "kline60" / "mark" / f"{s}.jsonl.gz"
        rows = [x for rec in read_raw(path) for x in json.loads(rec["body"])["result"]["list"]] if path.exists() else []
        df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close"]).astype({c: float for c in ("open", "high", "low", "close")})
        df["ts"] = pd.to_datetime(pd.to_numeric(df["ts"]), unit="ms", utc=True)
        out.append(df.drop_duplicates("ts").assign(symbol=s))
    return pd.concat(out, ignore_index=True)


def to_4h(h: pd.DataFrame, bars: pd.DatetimeIndex, syms: list[str]) -> dict[str, pd.DataFrame]:
    """Same rule as F2 clean.panels: a 4-hour bar exists only if all 4 hourly candles exist."""
    hours = pd.date_range(bars[0], bars[-1] + pd.Timedelta(hours=3), freq="h")
    wide = {c: h.pivot(index="ts", columns="symbol", values=c).reindex(index=hours, columns=syms) for c in ("open", "high", "low", "close")}
    g = lambda f: f.groupby(f.index.floor("4h"))
    full = g(wide["close"]).count().reindex(bars) == 4
    pan = {"mark_open": g(wide["open"]).first(), "mark_high": g(wide["high"]).max(),
           "mark_low": g(wide["low"]).min(), "mark_close": g(wide["close"]).last()}
    return {k: v.reindex(bars).where(full) for k, v in pan.items()} | {"mark_open_1h": wide["open"]}


def mark_panels(cfg: dict[str, Any]) -> None:
    bars = pd.read_parquet(CLEAN / "panel_close.parquet").index
    syms = list(universe(cfg)["symbol"])
    pan = to_4h(hourly_mark(syms), bars, syms)
    for k, v in pan.items():
        v.to_parquet(CLEAN / f"panel_{k}.parquet")
    close = pd.read_parquet(CLEAN / "panel_close.parquet")
    cov = pd.DataFrame({"last_bars": close.notna().sum(), "mark_bars": pan["mark_close"].notna().sum(),
                        "common_bars": (close.notna() & pan["mark_close"].notna()).sum()})
    cov.to_csv(CLEAN / "mark_coverage.csv")
    print(f"mark panels: common/last bars = {cov.common_bars.sum() / cov.last_bars.sum():.4f}; "
          f"symbols with <99% coverage: {list(cov.index[cov.common_bars < 0.99 * cov.last_bars])}")


def load_mark() -> dict[str, pd.DataFrame]:
    return {k: pd.read_parquet(CLEAN / f"panel_{k}.parquet") for k in ("mark_open", "mark_high", "mark_low", "mark_close", "mark_open_1h")}
