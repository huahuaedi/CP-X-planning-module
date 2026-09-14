"""Ensures this package's own internal modules can be imported.

Some code under ``opencda/planning_module`` still uses flat,
non-package-qualified imports (``from behavior_planner import ...``, ``from
MPC import MPC``, ``from utility import ...``).  The supported runtime enters
through the repository's ``opencda.py`` and qualified
``opencda.planning_module.*`` imports, so this package exposes its directory
as an import root until those internal imports are fully package-qualified.

This runs once, as early as possible: Python always executes a package's
``__init__.py`` before importing anything under it, so this is the single
place guaranteed to fix the path before any of this package's flat imports
can be reached.  The local guard in ``opencda_bridge/cpx_mpc_planner.py`` is
retained only for direct module imports and is a no-op in the normal path.
"""

import os
import sys

_PLANNING_MODULE_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PLANNING_MODULE_ROOT not in sys.path:
    sys.path.insert(0, _PLANNING_MODULE_ROOT)
