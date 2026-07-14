"""Small helpers for planner cache file management."""

from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path
from typing import Any, Dict, Optional, Union

CACHE_VERSION = 1


def compute_xodr_signature(xodr_path: Union[str, Path]) -> Dict[str, Any]:
    """Collect a stable signature for one OpenDRIVE file.

    input: `xodr_path` (`str | Path`)
    output: signature data (`Dict[str, object]`)
    """
    path = Path(xodr_path).resolve()
    content = path.read_bytes()
    stats = path.stat()
    return {
        "path": str(path),
        "sha1": hashlib.sha1(content).hexdigest(),
        "size": int(stats.st_size),
        "mtime_ns": int(stats.st_mtime_ns),
    }


def get_cache_paths(xodr_path: Union[str, Path], cache_root: Union[str, Path], signature: Dict[str, Any]) -> Dict[str, Path]:
    """Build the cache file paths for one OpenDRIVE input.

    input: `xodr_path` (`str | Path`), `cache_root` (`str | Path`), `signature` (`Dict[str, object]`)
    output: cache paths (`Dict[str, Path]`)
    """
    xodr_path = Path(xodr_path)
    cache_root = Path(cache_root)
    cache_key = f"{xodr_path.stem}_{signature['sha1'][:8]}"
    cache_directory = cache_root / cache_key
    adm_file = cache_directory / "map.adm"
    return {
        "directory": cache_directory,
        "adm_file": adm_file,
        "adm_config_file": cache_directory / "map.adm.txt",
        "planner_cache_file": cache_directory / "planner_cache.pkl",
        "metadata_file": cache_directory / "metadata.json",
    }


def load_metadata(path: Union[str, Path]) -> Optional[Dict[str, Any]]:
    """Read cache metadata if it exists and is valid JSON.

    input: `path` (`str | Path`)
    output: metadata or no result (`Dict[str, object] | None`)
    """
    path = Path(path)
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except json.JSONDecodeError:
        return None


def save_metadata(path: Union[str, Path], metadata: Dict[str, Any]) -> None:
    """Write cache metadata to disk.

    input: `path` (`str | Path`), `metadata` (`Dict[str, object]`)
    output: none (`None`)
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, sort_keys=True)


def load_pickle(path: Union[str, Path]) -> Any:
    """Read one pickle file from disk.

    input: `path` (`str | Path`)
    output: deserialized object (`object`)
    """
    with Path(path).open("rb") as file:
        return pickle.load(file)


def save_pickle(path: Union[str, Path], value: Any) -> None:
    """Write one pickle file to disk.

    input: `path` (`str | Path`), `value` (`object`)
    output: none (`None`)
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file:
        pickle.dump(value, file, protocol=pickle.HIGHEST_PROTOCOL)


def metadata_matches(
    metadata: Optional[Dict[str, Any]],
    signature: Dict[str, Any],
    centerline_spacing_m: float,
) -> bool:
    """Check whether saved metadata still matches the requested planner inputs.

    input: `metadata` (`Dict[str, object] | None`), `signature` (`Dict[str, object]`), `centerline_spacing_m` (`float`)
    output: whether the cache is reusable (`bool`)
    """
    if metadata is None:
        return False
    return (
        metadata.get("cache_version") == CACHE_VERSION
        and metadata.get("xodr_signature") == signature
        and float(metadata.get("centerline_spacing_m", -1.0)) == float(centerline_spacing_m)
    )
