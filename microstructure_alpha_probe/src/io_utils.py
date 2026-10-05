from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def session_id(symbol: str, now: datetime | None = None) -> str:
    stamp = (now or utc_now()).strftime("%Y-%m-%d_%H%M%S")
    return f"{stamp}_{symbol.lower()}"


def create_unique_dir(base: str | Path, name: str) -> Path:
    path = Path(base) / name
    path.mkdir(parents=True, exist_ok=False)
    return path


def atomic_write_json(data: Any, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(target)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=json_default), encoding="utf-8")
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_text(text: str, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(target)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_dataframe(frame: pd.DataFrame, path: str | Path, *, compression: str = "zstd") -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(target)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        suffix = target.suffix.lower()
        if suffix == ".parquet":
            frame.to_parquet(temporary, index=False, compression=compression)
        elif suffix == ".csv":
            frame.to_csv(temporary, index=False)
        else:
            raise ValueError(f"Unsupported dataframe extension: {target.suffix}")
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def json_default(value: Any) -> Any:
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.isoformat()
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dataset_fingerprint(paths: Iterable[str | Path]) -> dict[str, Any]:
    files = sorted(Path(p).resolve() for p in paths)
    rows = [{"path": str(p), "bytes": p.stat().st_size, "sha256": sha256_file(p)} for p in files]
    combined = hashlib.sha256()
    for row in rows:
        combined.update(json.dumps(row, sort_keys=True).encode("utf-8"))
    return {"sha256": combined.hexdigest(), "files": rows, "total_bytes": sum(r["bytes"] for r in rows)}


def git_commit(root: str | Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(Path(root)), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


def environment_manifest(packages: Iterable[str] = ()) -> dict[str, Any]:
    versions: dict[str, str | None] = {}
    for name in packages:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "packages": versions,
    }


def list_parquet_parts(root: str | Path) -> list[Path]:
    return sorted(Path(root).glob("part-*.parquet"))

