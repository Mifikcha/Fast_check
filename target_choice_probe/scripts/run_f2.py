"""Stage F2 command line. Run from target_choice_probe/: python scripts/run_f2.py <command>."""

from __future__ import annotations

import faulthandler
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import CONFIG, HOLDOUT_LOG, LOGS, WORK, holdout_open, load_config, now_utc, sha256_file  # noqa: E402


def freeze(cfg, digest):
    out = LOGS / "frozen_hash_F2.txt"
    if out.exists():
        print(out.read_text(encoding="utf-8"))
        raise SystemExit("already frozen: a change needs a new config version")
    LOGS.mkdir(parents=True, exist_ok=True)
    out.write_text(f"config: {CONFIG.name}\nfile_sha256: {sha256_file(CONFIG)}\nsection_sha256: {digest}\nfrozen_utc: {now_utc()}\n",
                   encoding="utf-8")
    print(out.read_text(encoding="utf-8"))


def check_frozen(digest):
    text = (LOGS / "frozen_hash_F2.txt").read_text(encoding="utf-8")
    if f"section_sha256: {digest}" not in text:
        raise SystemExit("config differs from the frozen one")


def open_holdout_F2(cfg, digest):
    check_frozen(digest)
    if holdout_open():
        raise SystemExit(f"holdout already opened:\n{HOLDOUT_LOG.read_text(encoding='utf-8')}")
    theta = json.loads((LOGS / "theta_selected_F2.json").read_text(encoding="utf-8"))  # must be frozen before
    entry = {"utc": now_utc(), "config_section_sha256": digest, "theta_selected": theta,
             "versions": {"features": json.loads((WORK / "features_meta.json").read_text(encoding="utf-8"))["n_kept"],
                          "code": {p.name: sha256_file(p) for p in sorted((CONFIG.parents[1] / "src").glob("*.py"))}}}
    HOLDOUT_LOG.write_text(json.dumps(entry, indent=1) + "\n", encoding="utf-8")
    print(HOLDOUT_LOG.read_text(encoding="utf-8"))


def main(cmd):
    sys.stdout.reconfigure(encoding="utf-8")
    faulthandler.enable()
    cfg, digest = load_config()
    if cmd == "freeze":
        return freeze(cfg, digest)
    check_frozen(digest)
    if cmd == "download":
        from src.download import download_all
        return download_all(cfg)
    if cmd == "clean":
        from src.clean import clean_all
        return clean_all(cfg)
    if cmd == "price_demo":
        from src.report import price_demo
        return price_demo(cfg)
    if cmd == "features":
        from src.features import build
        return build(cfg)
    if cmd == "walkforward":          # dev until the holdout is opened, then extends over the holdout
        from src.walkforward import run
        return run(cfg)
    if cmd == "portfolios":
        from src.portfolio import run
        return run(cfg)
    if cmd == "open_holdout_F2":
        return open_holdout_F2(cfg, digest)
    if cmd == "report":
        from src.report import write_all
        return write_all(cfg)
    raise SystemExit(f"unknown command {cmd}")


if __name__ == "__main__":
    main(sys.argv[1])
