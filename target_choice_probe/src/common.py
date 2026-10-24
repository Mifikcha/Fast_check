"""Stage F2 shared pieces: paths, frozen config, periods, holdout guard, attempts log, public REST client.

Same conventions as Stage F1 (funding_carry_probe/src/common.py), copied rather than imported so F1 stays untouched.
"""

from __future__ import annotations

import csv
import hashlib
import json
import random
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "stageF2_frozen.yaml"
RAW, CLEAN, LOGS, REPORTS = ROOT / "data" / "raw", ROOT / "data" / "clean", ROOT / "logs", ROOT / "reports"
WORK = ROOT / "data" / "work"          # feature memmap, IC series, predictions (not in git)
FIG = REPORTS / "figures"
HOLDOUT_LOG = LOGS / "holdout_opened_F2.log"
ATTEMPTS = LOGS / "attempts_F2.csv"


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_config(path: Path = CONFIG) -> tuple[dict[str, Any], str]:
    """Return (section, sha256 of the parsed section) — CRLF/LF rewrites do not change the hash."""
    section = yaml.safe_load(path.read_text(encoding="utf-8"))["stageF2"]
    canonical = json.dumps(section, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return section, hashlib.sha256(canonical.encode()).hexdigest()


def periods(cfg: dict[str, Any], months_of_data: float | None = None) -> dict[str, pd.Timestamp]:
    p = cfg["period"]
    end = pd.Timestamp(p["data_end_utc"])
    start = pd.Timestamp(p["analysis_start_utc"])
    if months_of_data is None:
        months_of_data = (end - start).days / 30.44
    days = p["holdout_days"] if months_of_data >= p["min_months_for_long_holdout"] else p["holdout_days_short"]
    hold = end - pd.Timedelta(days=days)
    emb = hold - pd.Timedelta(days=p["embargo_days"])
    return {"data_start": pd.Timestamp(p["data_start_utc"]), "start": start, "dev_end": emb,
            "holdout_start": hold, "end": end, "holdout_days": days}


def holdout_open() -> bool:
    return HOLDOUT_LOG.exists() and HOLDOUT_LOG.read_text(encoding="utf-8").strip() != ""


def last_usable(cfg: dict[str, Any]) -> pd.Timestamp:
    """Latest time any statistic may touch: dev end until open_holdout_F2 ran, then the data end."""
    p = periods(cfg)
    return p["end"] if holdout_open() else p["dev_end"]


def log_attempt(target: str, h: int, family: str, model: str, n_tests: int, period: str, note: str = "") -> None:
    new = not ATTEMPTS.exists()
    LOGS.mkdir(parents=True, exist_ok=True)
    with ATTEMPTS.open("a", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["utc", "target", "h", "feature_family", "model", "n_tests", "period", "note"])
        w.writerow([now_utc(), target, h, family, model, n_tests, period, note])


class Rest:
    """Public GET only: global rate limit, exponential backoff with jitter on 403/429/5xx/busy codes."""

    RETRY_CODES = {10006, 10016, 10429}

    def __init__(self, api: dict[str, Any]) -> None:
        self.api = api
        self.interval = 1.0 / float(api["max_requests_per_s"])
        self._next, self._lock = 0.0, threading.Lock()
        self.requests = 0

    def _wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            delay = max(0.0, self._next - now)
            self._next = max(now, self._next) + self.interval
        time.sleep(delay)

    def get(self, path: str, params: dict[str, Any]) -> str:
        url = f"{self.api['rest_base']}{path}?{urllib.parse.urlencode(params)}"
        backoff = float(self.api["backoff_initial_s"])
        for attempt in range(int(self.api["max_retries"]) + 1):
            self._wait()
            self.requests += 1
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "target-choice-probe/F2"})
                with urllib.request.urlopen(req, timeout=float(self.api["timeout_s"])) as r:
                    body = r.read().decode()
                if json.loads(body).get("retCode") not in self.RETRY_CODES:
                    return body
            except (urllib.error.URLError, TimeoutError, ConnectionError, json.JSONDecodeError) as exc:
                if isinstance(exc, urllib.error.HTTPError) and exc.code == 403:
                    backoff = max(backoff, 600.0)      # docs: IP ban, stay off >= 10 min
            time.sleep(min(backoff, float(self.api["backoff_max_s"]) if backoff < 600 else backoff)
                       * (1 + random.random() * 0.2))
            backoff *= 2
        raise RuntimeError(f"gave up after retries: {url}")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def md(df: pd.DataFrame, index: bool = True, digits: int = 4) -> str:
    """Markdown table without the optional `tabulate` dependency (the shared .venv does not have it)."""
    d = df.reset_index() if index else df
    fmt = lambda v: f"{v:.{digits}g}" if isinstance(v, float) else str(v)
    lines = ["| " + " | ".join(map(str, d.columns)) + " |", "|" + "---|" * len(d.columns)]
    return "\n".join(lines + ["| " + " | ".join(fmt(v) for v in row) + " |" for row in d.itertuples(index=False)])
