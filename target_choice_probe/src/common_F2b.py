"""Stage F2b shared pieces: frozen F2b config, attempts log. F2 modules are imported read only."""

from __future__ import annotations

import csv
import hashlib
import json
from typing import Any

import yaml

from .common import LOGS, ROOT, WORK, now_utc

CONFIG_F2B = ROOT / "configs" / "stageF2b_frozen.yaml"
ATTEMPTS_F2B = LOGS / "attempts_F2b.csv"
FROZEN_F2B = LOGS / "frozen_hash_F2b.txt"
WORK_F2B = WORK / "f2b"
HOLDOUT_NOTE = "контроль уже открывался в F2: подтверждаем только знак и порядок величины"


def load_f2b() -> tuple[dict[str, Any], str]:
    section = yaml.safe_load(CONFIG_F2B.read_text(encoding="utf-8"))["stageF2b"]
    canonical = json.dumps(section, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return section, hashlib.sha256(canonical.encode()).hexdigest()


def log_attempt_f2b(question: str, what: str, n_tests: int, period: str, note: str = "") -> None:
    new = not ATTEMPTS_F2B.exists()
    LOGS.mkdir(parents=True, exist_ok=True)
    with ATTEMPTS_F2B.open("a", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["utc", "question", "what", "n_tests", "period", "note"])
        w.writerow([now_utc(), question, what, n_tests, period, note])
