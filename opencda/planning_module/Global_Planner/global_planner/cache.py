"""Small cache helpers for the custom OpenDRIVE global planner."""

from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path
from typing import Any, Mapping


CACHE_VERSION = 1


def compute_xodr_signature(xodr_path: str | Path) -> dict[str, object]:
    """Return a stable signature for one OpenDRIVE file."""

    path = Path(xodr_path).resolve()
    stat = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": str(path),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": digest.hexdigest(),
    }


def get_cache_paths(
    xodr_path: str | Path,
    cache_root: str | Path,
    signature: Mapping[str, object],
) -> dict[str, Path]:
    """Build cache artifact paths for one map signature."""

    root = Path(cache_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    stem = Path(xodr_path).stem
    digest = str(signature.get("sha256", ""))[:12] or "unknown"
    prefix = f"{stem}_{digest}"
    return {
        "metadata_file": root / f"{prefix}.metadata.json",
        "planner_cache_file": root / f"{prefix}.planner.pkl",
        "adm_file": root / f"{prefix}.adm",
        "adm_config_file": root / f"{prefix}.adm",
    }


def load_metadata(path: str | Path) -> dict[str, object]:
    try:
        with Path(path).open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def save_metadata(path: str | Path, metadata: Mapping[str, object]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as f:
        json.dump(dict(metadata), f, indent=2, sort_keys=True)


def metadata_matches(
    metadata: Mapping[str, object],
    signature: Mapping[str, object],
    centerline_spacing_m: float,
) -> bool:
    return (
        int(metadata.get("cache_version", -1)) == int(CACHE_VERSION)
        and float(metadata.get("centerline_spacing_m", -1.0)) == float(centerline_spacing_m)
        and dict(metadata.get("xodr_signature", {}) or {}) == dict(signature or {})
    )


def load_pickle(path: str | Path) -> Any:
    with Path(path).open("rb") as f:
        return pickle.load(f)


def save_pickle(path: str | Path, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as f:
        pickle.dump(value, f, protocol=pickle.HIGHEST_PROTOCOL)
