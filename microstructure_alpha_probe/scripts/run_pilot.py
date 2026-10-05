"""Single Windows-friendly entry point for the feasibility probe."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from src.build_dataset import build_dataset  # noqa: E402
from src.collect import collect_blocking  # noqa: E402
from src.config import load_config, project_path  # noqa: E402
from src.evaluate import analyze_dataset  # noqa: E402
from src.report import render_report  # noqa: E402
from src.long_collection import collect_long, experiment_status, resolve_experiment  # noqa: E402
from src.long_pipeline import run_daily  # noqa: E402
from src.long_report import analyze_long, render_long_report  # noqa: E402
import yaml


def _positive_float(value: str) -> float:
    number = float(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return number


def _duration(seconds: Any) -> str:
    if seconds is None:
        return "до Ctrl+C"
    value = max(0, int(float(seconds)))
    hours, remainder = divmod(value, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _bytes(value: Any) -> str:
    size = float(value or 0)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.1f} {unit}"
        size /= 1024
    return "n/a"


def _progress(item: dict[str, Any]) -> None:
    timestamp = datetime.now().strftime("%H:%M:%S")
    phase = str(item.get("phase", "work")).upper()
    if phase == "COLLECT":
        statuses = {
            "starting": "запуск", "connecting": "подключение", "connected": "подключено",
            "subscribed": "данные идут", "reconnecting": "переподключение",
            "complete": "завершено", "failed": "ошибка", "cancelled": "отменено",
        }
        status = statuses.get(str(item.get("status")), str(item.get("status")))
        percent = "" if item.get("percent") is None else f" | {item['percent']:.2f}%"
        last_age = item.get("last_message_age_s")
        last = "ожидание первого сообщения" if last_age is None else f"последнее msg {last_age:.1f}s назад"
        print(
            f"[{timestamp}] COLLECT {status} | прошло {_duration(item.get('elapsed_s'))}"
            f" | осталось {_duration(item.get('remaining_s'))}{percent} | wire-msg {item.get('wire_messages', 0):,}"
            f" ({item.get('message_rate_hz', 0):.1f}/s) | book-msg {item.get('book_messages', 0):,}"
            f" | сделок {item.get('trades', 0):,} | ticker {item.get('ticker_messages', 0):,}"
            f" | сохранено {item.get('published_parts', 0)} parts / {_bytes(item.get('saved_bytes'))}"
            f" | в буфере {item.get('buffered_rows', 0):,} строк | стакан {'OK' if item.get('book_valid') else 'UNSYNCED'}"
            f" | reconnects {item.get('reconnects', 0)} | {last}",
            flush=True,
        )
        return
    print(
        f"[{timestamp}] {phase} [{item.get('step', '?')}/{item.get('total', '?')}]"
        f" {item.get('message', '')} | прошло {_duration(item.get('elapsed_s'))}",
        flush=True,
    )


def _latest_child(root: Path, *, required: str | None = None) -> Path:
    candidates = [path for path in root.iterdir() if path.is_dir() and (required is None or (path / required).exists())] if root.exists() else []
    if not candidates:
        raise FileNotFoundError(f"no usable sessions/runs under {root}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _capture(config: dict, requested: str | None) -> str:
    if requested:
        return requested
    processed = project_path(config, "processed_dir")
    if processed.exists():
        ready = [path for path in processed.iterdir() if path.is_dir() and (path / "research_table.parquet").exists()]
        if ready:
            return max(ready, key=lambda path: path.stat().st_mtime).name
    return _latest_child(project_path(config, "raw_dir") / "sessions", required="session_end.json").name


def _print_result(run: Path, verdict: dict) -> None:
    import json
    dataset = json.loads((run / "dataset_manifest.json").read_text(encoding="utf-8"))
    quality = json.loads((run / "data_quality.json").read_text(encoding="utf-8"))
    values = verdict.get("values", {})
    print(f"Run directory: {run}")
    print(f"Dataset rows: {dataset.get('rows')} · usable book: {dataset.get('book_usable_pct')}%")
    print(f"Data quality: {quality.get('status')} · {', '.join(quality.get('reasons', [])) or 'no configured warnings'}")
    print(f"Primary test rank IC: {values.get('test_rank_ic')}")
    print(f"Primary confirmatory-cost net PnL: {values.get('confirmatory_cost_net_pnl')}")
    print(f"Verdict: {verdict['verdict']}")
    print(f"Report: {run / 'report.md'}")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Public BTCUSDT microstructure alpha feasibility probe")
    parser.add_argument("--config", type=Path, default=PROJECT / "configs" / "pilot.yaml")
    commands = parser.add_subparsers(dest="command", required=True)
    collect = commands.add_parser("collect", help="record immutable public Bybit data")
    collect.add_argument("--hours", type=_positive_float, help="duration; omit to run until Ctrl+C")
    collect.add_argument("--days", type=int, help="complete UTC days (7 minimum; 10–14 preferred)")
    collect.add_argument("--resume", action="store_true", help="new immutable session in the existing experiment")
    collect.add_argument("--experiment-id", help="experiment to create or resume")
    collect.add_argument("--smoke-minutes", type=_positive_float, help="bounded live smoke; not a multi-day run")
    collect.add_argument("--progress-seconds", type=_positive_float, default=30, help="console update interval (default: 30)")
    build = commands.add_parser("build", help="reconstruct one raw session")
    build.add_argument("--session")
    analyze = commands.add_parser("analyze", help="build if needed, then run the full analysis")
    analyze.add_argument("--session")
    analyze.add_argument("--dataset", type=Path)
    report = commands.add_parser("report", help="regenerate report.md from saved tables")
    report.add_argument("--run", required=True)
    status = commands.add_parser("status", help="show long experiment health")
    status.add_argument("--experiment-id")
    daily = commands.add_parser("daily", help="analyze completed UTC days")
    daily.add_argument("--experiment-id")
    daily.add_argument("--include-partial", action="store_true", help="smoke only; excluded from verdict")
    long_analysis = commands.add_parser("analyze-long", help="daily + longitudinal analysis")
    long_analysis.add_argument("--experiment-id")
    long_analysis.add_argument("--include-partial", action="store_true", help="smoke only")
    long_report = commands.add_parser("report-long", help="regenerate plots and report from saved tables")
    long_report.add_argument("--experiment-id")
    long_report.add_argument("--analysis", type=Path)
    everything = commands.add_parser("all", help="collect, build, and analyze")
    everything.add_argument("--hours", type=_positive_float, required=True)
    everything.add_argument("--progress-seconds", type=_positive_float, default=30, help="console update interval (default: 30)")
    args = parser.parse_args()
    config = load_config(args.config)

    if args.command in {"status", "daily", "analyze-long", "report-long"}:
        manifest = resolve_experiment(args.experiment_id)
        if args.command == "status":
            item = experiment_status(manifest, project_path(config, "raw_dir"))
            for key, value in item.items():
                print(f"{key}: {value}")
            return
        if args.command == "daily":
            for path in run_daily(config, manifest, include_partial=args.include_partial, progress=_progress):
                print(path / "daily_report.md")
            return
        if args.command == "analyze-long":
            path = analyze_long(config, manifest, include_partial=args.include_partial, progress=_progress)
            print(path / "longitudinal_report.md")
            return
        root = project_path(config, "runs_dir") / f"long_{manifest.parent.name}" / "analyses"
        analysis = args.analysis or _latest_child(root, required="longitudinal_summary.json")
        print(render_long_report(analysis))
        return

    if args.command == "collect":
        if args.days is not None and args.days < 7:
            parser.error("--days must be at least 7")
        if args.hours is not None and (args.days is not None or args.resume or args.smoke_minutes):
            parser.error("--hours is the legacy pilot mode and cannot be mixed with long collection")
        if args.hours is None:
            frozen = yaml.safe_load((PROJECT / "configs" / "confirmatory_ridge30.yaml").read_text(encoding="utf-8"))
            path = collect_long(config, frozen, days=args.days, resume=args.resume,
                                experiment_id=args.experiment_id, progress=_progress,
                                progress_interval_s=args.progress_seconds,
                                smoke_seconds=args.smoke_minutes * 60 if args.smoke_minutes else None)
            print(f"Experiment: {path.parent.name}\nManifest: {path}")
            return
        duration = None if args.hours is None else args.hours * 3600
        path = collect_blocking(
            config, duration_s=duration, progress_callback=_progress,
            progress_interval_s=args.progress_seconds,
        )
        print(f"Capture session: {path.name}")
        print(f"Manifest: {path / 'session_end.json'}")
        return
    if args.command == "report":
        candidate = Path(args.run)
        run = candidate if candidate.is_absolute() else project_path(config, "runs_dir") / candidate
        print(f"[{datetime.now():%H:%M:%S}] REPORT Regenerating {run / 'report.md'}", flush=True)
        print(render_report(run))
        return
    if args.command == "all":
        capture = collect_blocking(
            config, duration_s=args.hours * 3600, progress_callback=_progress,
            progress_interval_s=args.progress_seconds,
        ).name
        dataset = build_dataset(config, capture, progress=_progress)
        run, verdict = analyze_dataset(config, dataset, capture=capture, progress=_progress)
        _print_result(run, verdict)
        return
    if args.command == "build":
        capture = _capture(config, args.session)
        print(build_dataset(config, capture, progress=_progress))
        return

    capture = _capture(config, args.session)
    dataset = args.dataset.resolve() if args.dataset else project_path(config, "processed_dir") / capture / "research_table.parquet"
    if not dataset.exists():
        dataset = build_dataset(config, capture, progress=_progress)
    run, verdict = analyze_dataset(config, dataset, capture=capture, progress=_progress)
    _print_result(run, verdict)


if __name__ == "__main__":
    main()
