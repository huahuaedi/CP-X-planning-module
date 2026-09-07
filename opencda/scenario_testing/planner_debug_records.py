"""Canonical reader for CP-X planner experiment records.

New experiments write JSONL only. CSV remains a compatibility input for old
runs, but report scripts should not each implement their own parser.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping


def load_planner_records(debug_dir: Path) -> List[Dict[str, Any]]:
    """Load one planner run, preferring the complete JSONL record."""

    debug_dir = Path(debug_dir)
    jsonl_path = debug_dir / "opencda_planner_debug.jsonl"
    if jsonl_path.is_file():
        records = []
        with jsonl_path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    value = json.loads(line)
                except ValueError as exc:
                    raise ValueError(
                        "invalid planner JSONL %s:%d: %s"
                        % (jsonl_path, line_number, exc)
                    )
                if isinstance(value, Mapping):
                    records.append(dict(value))
        return records

    csv_path = debug_dir / "opencda_planner_debug.csv"
    if csv_path.is_file():
        with csv_path.open(newline="", encoding="utf-8") as stream:
            return [dict(row) for row in csv.DictReader(stream)]
    return []


def number(record: Mapping[str, Any], key: str, default: float = math.nan) -> float:
    try:
        value = float(record.get(key, ""))
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def flag(record: Mapping[str, Any], key: str) -> bool:
    value = record.get(key, False)
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "1.0", "true", "yes"}
