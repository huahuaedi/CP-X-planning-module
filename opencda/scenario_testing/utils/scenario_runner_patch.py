# -*- coding: utf-8 -*-
# License: TDG-Attribution-NonCommercial-NoDistrib
import glob
import importlib
import inspect
import os
import sys


def patch_scenario_runner_imports(sr):
    """
    Patch ScenarioRunner to skip osc2_scenario imports when OSC2 deps are missing.
    """
    if getattr(sr.ScenarioRunner, "_opencda_ignore_import_errors", False):
        return

    def _get_scenario_class_or_fail(self, scenario):
        scenarios_list = glob.glob(
            "{}/srunner/scenarios/*.py".format(os.getenv('SCENARIO_RUNNER_ROOT', "./"))
        )
        scenarios_list.append(self._args.additionalScenario)

        for scenario_file in scenarios_list:
            if not scenario_file:
                continue
            module_name = os.path.basename(scenario_file).split('.')[0]
            sys.path.insert(0, os.path.dirname(scenario_file))
            try:
                scenario_module = importlib.import_module(module_name)
            except Exception:
                if module_name == "osc2_scenario":
                    scenario_module = None
                else:
                    raise
            finally:
                sys.path.pop(0)
            if scenario_module is None:
                continue

            for member in inspect.getmembers(scenario_module, inspect.isclass):
                if scenario in member:
                    return member[1]

        print("Scenario '{}' not supported ... Exiting".format(scenario))
        sys.exit(-1)

    sr.ScenarioRunner._get_scenario_class_or_fail = _get_scenario_class_or_fail
    sr.ScenarioRunner._opencda_ignore_import_errors = True
