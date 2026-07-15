import csv
import json
import os
import tempfile
import unittest

from utility.artifacts import write_dict_csv_artifact, write_json_artifact


class ArtifactWriterTests(unittest.TestCase):
    def test_write_dict_csv_artifact_preserves_fields_and_fills_missing_values(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = os.path.join(tmp_dir, "trace.csv")
            write_dict_csv_artifact(
                path,
                rows=[
                    {"a": 1, "b": 2, "extra": 3},
                    {"a": 4},
                ],
                fieldnames=["a", "b"],
            )

            with open(path, "r", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))

        self.assertEqual(list(rows[0].keys()), ["a", "b"])
        self.assertEqual(rows[0]["a"], "1")
        self.assertEqual(rows[0]["b"], "2")
        self.assertEqual(rows[1]["a"], "4")
        self.assertEqual(rows[1]["b"], "")

    def test_write_json_artifact_creates_parent_directory(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = os.path.join(tmp_dir, "nested", "summary.json")
            write_json_artifact(path, {"b": 2, "a": 1})

            with open(path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)

        self.assertEqual(payload, {"a": 1, "b": 2})


if __name__ == "__main__":
    unittest.main()
