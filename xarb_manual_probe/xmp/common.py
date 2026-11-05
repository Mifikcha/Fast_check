"""Config access, hashes and the attempts log."""
import csv
import datetime as dt
import hashlib
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "stageX2_frozen.yaml"
LOGS = ROOT / "logs"
FROZEN = LOGS / "frozen_hash.txt"
ATTEMPTS = LOGS / "attempts.csv"
HOLDOUT = LOGS / "holdout_opened.json"
SELECTED = LOGS / "selected_variant.json"
DERIVED = ROOT / "data" / "derived"


def now_utc():
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def load_config(path=CONFIG):
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))["stageX2"]


def section_hash(path=CONFIG):
    """Hash of the parsed section: comments may change, rules may not."""
    return hashlib.sha256(json.dumps(load_config(path), sort_keys=True).encode()).hexdigest()


def attempt(action, reason, digest):
    new = not ATTEMPTS.exists()
    LOGS.mkdir(exist_ok=True)
    with ATTEMPTS.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["utc", "action", "section_sha256", "reason"])
        w.writerow([now_utc(), action, digest, reason])


def frozen_digest():
    """Latest frozen hash; every line of the file is one version."""
    if not FROZEN.exists():
        return None
    return FROZEN.read_text(encoding="utf-8").strip().splitlines()[-1].split()[1]


def freeze(reason):
    digest = section_hash()
    if frozen_digest() == digest:
        raise SystemExit(f"already frozen: {digest}")
    if frozen_digest() and not reason:
        raise SystemExit("config changed after the freeze: a new version needs --reason")
    if HOLDOUT.exists():
        raise SystemExit("holdout is open: the rules can no longer change")
    with FROZEN.open("a", encoding="utf-8") as f:
        f.write(f"{now_utc()} {digest}\n")
    attempt("freeze", reason or "first version", digest)
    return digest


def require_frozen():
    digest = section_hash()
    if digest != frozen_digest():
        raise SystemExit("config differs from the frozen hash: run freeze --reason first")
    return digest
