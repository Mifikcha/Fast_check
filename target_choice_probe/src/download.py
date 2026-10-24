"""Resumable raw download from the public Bybit REST API (no keys).

Raw files are immutable: one file = one symbol/series (a .jsonl.gz of the API answers exactly as received),
written once, atomically, and skipped if it already exists. A re-run fills only what is missing.
"""

from __future__ import annotations

import gzip
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd

from .common import RAW, ROOT, Rest, now_utc, periods, sha256_file

KLINE = {"last": "/v5/market/kline", "premium": "/v5/market/premium-index-price-kline"}
HOUR_MS = 3_600_000


def _write_once(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    with gzip.open(tmp, "wt", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def _call(rest: Rest, path: str, params: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    body = rest.get(path, params)
    parsed = json.loads(body)
    if parsed.get("retCode") != 0:
        raise RuntimeError(f"{path} {params}: {parsed.get('retCode')} {parsed.get('retMsg')}")
    return {"path": path, "params": params, "received_utc": now_utc(), "body": body}, parsed["result"]


def read_raw(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def universe(cfg: dict[str, Any]) -> pd.DataFrame:
    path = ROOT / cfg["universe"]["stage4_universe_csv"]
    if sha256_file(path) != cfg["universe"]["stage4_universe_sha256"]:
        raise RuntimeError(f"{path} differs from the frozen Stage 4 snapshot")
    u = pd.read_csv(path)
    return u[u["turnover_rank"].notna()].sort_values("turnover_rank").reset_index(drop=True)


def instruments(rest: Rest) -> None:
    path = RAW / "instruments" / "linear.jsonl.gz"
    if path.exists():
        return
    recs, cursor = [], ""
    while True:
        params = {"category": "linear", "limit": 1000} | ({"cursor": cursor} if cursor else {})
        rec, res = _call(rest, "/v5/market/instruments-info", params)
        recs.append(rec)
        cursor = res.get("nextPageCursor") or ""
        if not cursor:
            break
    _write_once(path, recs)


def windows(start_ms: int, end_ms: int, n: int) -> list[tuple[int, int]]:
    """Inclusive [lo, hi] windows of n hourly points covering [start_ms, end_ms)."""
    out, t = [], start_ms
    while t < end_ms:
        hi = min(t + (n - 1) * HOUR_MS, end_ms - HOUR_MS)
        out.append((t, hi))
        t = hi + HOUR_MS
    return out


def kline(rest: Rest, series: str, sym: str, start_ms: int, end_ms: int, limit: int) -> None:
    path = RAW / "kline60" / series / f"{sym}.jsonl.gz"
    if path.exists():
        return
    recs = [_call(rest, KLINE[series], {"category": "linear", "symbol": sym, "interval": 60,
                                        "start": lo, "end": hi, "limit": limit})[0]
            for lo, hi in windows(start_ms, end_ms, limit)]
    _write_once(path, recs)


def open_interest(rest: Rest, sym: str, start_ms: int, end_ms: int, limit: int) -> None:
    path = RAW / "oi1h" / f"{sym}.jsonl.gz"
    if path.exists():
        return
    recs = [_call(rest, "/v5/market/open-interest", {"category": "linear", "symbol": sym, "intervalTime": "1h",
                                                     "startTime": lo, "endTime": hi, "limit": limit})[0]
            for lo, hi in windows(start_ms, end_ms, limit)]
    _write_once(path, recs)


def funding_history(rest: Rest, sym: str, start_ms: int, end_ms: int, limit: int) -> None:
    path = RAW / "funding" / f"{sym}.jsonl.gz"
    if path.exists():
        return
    recs, end = [], end_ms
    while True:  # docs: startTime alone is an error; endTime alone pages back `limit` records at a time
        rec, res = _call(rest, "/v5/market/funding/history", {"category": "linear", "symbol": sym, "endTime": end, "limit": limit})
        recs.append(rec)
        if not res["list"]:
            break
        end = min(int(x["fundingRateTimestamp"]) for x in res["list"]) - 1
        if end < start_ms - 30 * 24 * HOUR_MS:     # one month before the data start is enough for the interval
            break
    _write_once(path, recs)


def download_all(cfg: dict[str, Any], log=print) -> None:
    rest = Rest(cfg["api"])
    instruments(rest)
    uni = universe(cfg)
    p = periods(cfg)
    s_ms, e_ms = int(p["data_start"].value // 10**6), int(p["end"].value // 10**6)
    a = cfg["api"]
    jobs = []
    for sym in uni["symbol"]:
        jobs.append((funding_history, (rest, sym, s_ms, e_ms - 1, a["funding_limit"])))
        jobs += [(kline, (rest, s, sym, s_ms, e_ms, a["kline_limit"])) for s in KLINE]
        jobs.append((open_interest, (rest, sym, s_ms, e_ms, a["oi_limit"])))
    log(f"{now_utc()} universe {len(uni)} symbols, {len(jobs)} files")
    done = failed = 0
    with ThreadPoolExecutor(int(a["threads"])) as pool:
        futures = {pool.submit(fn, *args): (fn.__name__, args[1:3]) for fn, args in jobs}
        for fut in as_completed(futures):
            try:
                fut.result()
                done += 1
            except Exception as exc:  # keep going; a re-run fills what is missing
                failed += 1
                log(f"FAILED {futures[fut]}: {exc}")
            if (done + failed) % 40 == 0:
                log(f"{now_utc()} files {done + failed}/{len(jobs)} failed={failed} requests={rest.requests}")
    log(f"{now_utc()} download finished: ok={done} failed={failed} requests={rest.requests}")
