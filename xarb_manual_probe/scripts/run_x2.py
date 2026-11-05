"""Stage X2A command line. Run from xarb_manual_probe/: python scripts/run_x2.py <command>."""
import argparse
import logging
import socket
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from xmp import common  # noqa: E402
from xmp.common import LOGS, load_config  # noqa: E402


def record(a):
    from xmp.recorder import Recorder
    if (LOGS / "STOPPED_DISK").exists():
        raise SystemExit(f"stopped for disk reserve: {(LOGS / 'STOPPED_DISK').read_text(encoding='utf-8')}")
    lock = socket.socket()
    try:
        lock.bind(("127.0.0.1", 47823))  # the scheduler starts this every 5 minutes; a second copy leaves at once
    except OSError:
        raise SystemExit("already recording")
    LOGS.mkdir(exist_ok=True)
    logging.basicConfig(filename=LOGS / "recorder.log", level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    Recorder(load_config()).run(a.seconds)


def dev(a):
    from xmp import evaluate
    print(evaluate.dev(load_config()))


def open_holdout(a):
    from xmp import evaluate
    print(evaluate.open_holdout(load_config()))


def holdout(a):
    from xmp import evaluate
    print(len(evaluate.holdout(load_config())))


def weekly(a):
    from xmp import report
    print(report.weekly(load_config()))


def verdict(a):
    from xmp import report
    print(report.verdict(load_config()))


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("record")
    r.add_argument("--seconds", type=float)
    r.set_defaults(fn=record)
    f = sub.add_parser("freeze")
    f.add_argument("--reason", default="")
    f.set_defaults(fn=lambda a: print(common.freeze(a.reason)))
    for name, fn in (("dev", dev), ("open-holdout", open_holdout), ("holdout", holdout), ("weekly", weekly), ("verdict", verdict)):
        sub.add_parser(name).set_defaults(fn=fn)
    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
