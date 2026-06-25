import importlib.util
import os
import sys
import tempfile
import types
import unittest
from unittest import mock


def _script_path() -> str:
    tests_dir = os.path.dirname(os.path.abspath(__file__))
    planning_module_dir = os.path.dirname(tests_dir)
    opencda_dir = os.path.dirname(planning_module_dir)
    project_root = os.path.dirname(opencda_dir)
    return os.path.join(project_root, "scripts", "netconvert_carla.py")


def _load_netconvert_module():
    fake_etree = types.ModuleType("lxml.etree")
    fake_lxml = types.ModuleType("lxml")
    fake_lxml.etree = fake_etree
    fake_sumolib = types.ModuleType("sumolib")
    fake_sumolib.net = types.SimpleNamespace(readNet=lambda *_args, **_kwargs: None)

    spec = importlib.util.spec_from_file_location(
        "netconvert_carla_test_module",
        _script_path(),
    )
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(
        os.environ,
        {"SUMO_HOME": os.environ.get("SUMO_HOME", "/usr/share/sumo")},
        clear=False,
    ):
        with mock.patch.dict(
            sys.modules,
            {
                "lxml": fake_lxml,
                "lxml.etree": fake_etree,
                "sumolib": fake_sumolib,
            },
            clear=False,
        ):
            spec.loader.exec_module(module)
    return module


class NetconvertCarlaTests(unittest.TestCase):
    def test_falls_back_to_plain_netconvert_output_when_carla_import_is_unavailable(self):
        module = _load_netconvert_module()

        with tempfile.TemporaryDirectory() as tmp_dir:
            xodr_path = os.path.join(tmp_dir, "Town06.xodr")
            output_path = os.path.join(tmp_dir, "Town06.net.xml")
            with open(xodr_path, "w", encoding="utf-8") as handle:
                handle.write("<OpenDRIVE />")

            def _fake_subprocess_call(command):
                tmp_net_path = command[command.index("--output-file") + 1]
                with open(tmp_net_path, "w", encoding="utf-8") as handle:
                    handle.write("<net fallback='true' />")
                return 0

            with mock.patch.object(module, "carla", None):
                with mock.patch.object(
                    module,
                    "_CARLA_IMPORT_ERROR",
                    TypeError("'carla.libcarla' is not a package"),
                ):
                    with mock.patch.object(module.subprocess, "call", side_effect=_fake_subprocess_call):
                        module._netconvert_carla_impl(
                            xodr_file=xodr_path,
                            output=output_path,
                            tmpdir=tmp_dir,
                        )

            with open(output_path, "r", encoding="utf-8") as handle:
                self.assertEqual(handle.read(), "<net fallback='true' />")


if __name__ == "__main__":
    unittest.main()
