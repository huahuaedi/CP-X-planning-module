"""Ensures this package's own internal modules can be imported.

Most code under ``opencda/planning_module`` uses flat, non-package-qualified
imports (``from behavior_planner import ...``, ``from MPC import MPC``,
``from utility import ...``) because it also runs standalone via
``python main.py <scenario>`` from inside this directory, where the
directory itself is the import root. That assumption breaks when this code
is instead reached through the qualified ``opencda.planning_module.*`` path
(e.g. ``VehicleManager`` -> ``opencda_bridge.cpx_mpc_planner`` -> ``pipeline``
-> ``behavior_planner``), since ``opencda/planning_module`` is then never
added to ``sys.path`` on its own.

This runs once, as early as possible: Python always executes a package's
``__init__.py`` before importing anything under it, so this is the single
place guaranteed to fix the path before any of this package's flat imports
can be reached -- rather than relying on each entry point (e.g.
``opencda_bridge/cpx_mpc_planner.py``, ``opencda/scenario_testing/utils/
cpx_scenario_bridge.py``) to remember to call its own copy of this fix
early enough. Those local copies are left in place as defense in depth
(e.g. a module imported directly by file path, bypassing this __init__), but
are redundant no-ops in the normal import path once this has already run.
"""

import os
import sys

_PLANNING_MODULE_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PLANNING_MODULE_ROOT not in sys.path:
    sys.path.insert(0, _PLANNING_MODULE_ROOT)
