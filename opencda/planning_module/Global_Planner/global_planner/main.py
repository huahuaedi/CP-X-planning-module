"""Small direct-run entry point for the global planner package."""

from __future__ import annotations

import sys
from pathlib import Path

# Allow `python main.py` from inside `global_planner/` by exposing the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from global_planner import GlobalPlanner

XODR_PATH = PROJECT_ROOT / "maps" / "Town10HD_Opt.xodr"
CACHE_ROOT = PROJECT_ROOT / "cache"
START_POINT = {"x": -86.8, "y": 133.5, "z": 0.0}
GOAL_POINT = {"x": 59.4, "y": 137.8, "z": 0.0}


def main() -> None:
    """Run one simple planner query and print the route length.

    input: none (`None`)
    output: none (`None`)
    """
    planner = GlobalPlanner(XODR_PATH, cache_root=CACHE_ROOT)
    planner.load()
    try:
        route = planner.trace_route(START_POINT, GOAL_POINT)
        print(route.length_m)
    finally:
        planner.close()


if __name__ == "__main__":
    main()
