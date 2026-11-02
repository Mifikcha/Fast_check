"""Stage F2b command line. Run from target_choice_probe/: python scripts/run_f2b.py <command>."""

from __future__ import annotations

import faulthandler
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import load_config, now_utc, sha256_file  # noqa: E402
from src.common_F2b import CONFIG_F2B, FROZEN_F2B, load_f2b  # noqa: E402


def main(cmd):
    sys.stdout.reconfigure(encoding="utf-8")
    faulthandler.enable()
    cfg, f2_digest = load_config()
    fb, digest = load_f2b()
    if cmd == "freeze":
        if FROZEN_F2B.exists():
            raise SystemExit("already frozen: a change needs a new config version")
        FROZEN_F2B.write_text(f"config: {CONFIG_F2B.name}\nfile_sha256: {sha256_file(CONFIG_F2B)}\nsection_sha256: {digest}\n"
                              f"f2_section_sha256: {f2_digest}\nfrozen_utc: {now_utc()}\n", encoding="utf-8")
        return print(FROZEN_F2B.read_text(encoding="utf-8"))
    if f"section_sha256: {digest}" not in FROZEN_F2B.read_text(encoding="utf-8"):
        raise SystemExit("F2b config differs from the frozen one")
    if cmd == "download_mark":
        from src.data_F2b import download_mark
        return download_mark(cfg)
    if cmd == "mark_panels":
        from src.data_F2b import mark_panels
        return mark_panels(cfg)
    if cmd == "q2":
        from src.q2_F2b import run
        return run(cfg, fb)
    if cmd == "q2_report":
        from src.q2_F2b import report
        return report(cfg, fb)
    if cmd == "pm_features":
        from src.q1_F2b import pm_features
        return pm_features(cfg, fb)
    if cmd == "pm_walkforward":
        from src.q1_F2b import pm_walkforward
        return pm_walkforward(cfg, fb)
    if cmd in ("q1", "q3", "q4", "q5", "corrections", "verdict"):
        import importlib
        mod = importlib.import_module("src.q1_F2b" if cmd == "q1" else "src.strategies_F2b" if cmd != "verdict" else "src.verdict_F2b")
        return getattr(mod, f"run_{cmd}")(cfg, fb)
    raise SystemExit(f"unknown command {cmd}")


if __name__ == "__main__":
    main(sys.argv[1])
