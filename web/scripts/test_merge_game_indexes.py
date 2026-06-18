#!/usr/bin/env python3
"""Tests for merge-game-indexes.py (hyphenated module loaded via importlib).

Run directly with: python3 web/scripts/test_merge_game_indexes.py
"""
import importlib.util
import json
import os
import tempfile
import unittest

_MODULE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "merge-game-indexes.py")
_spec = importlib.util.spec_from_file_location("merge_game_indexes", _MODULE_PATH)
mgi = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mgi)


class MergeVisualizersTest(unittest.TestCase):
    def test_none_existing(self):
        self.assertEqual(mgi.merge_visualizers(None, ["v2"]), ["v2"])

    def test_legacy_array_format(self):
        existing = json.dumps(["default"])
        self.assertEqual(mgi.merge_visualizers(existing, ["v2"]), ["default", "v2"])

    def test_object_format(self):
        existing = json.dumps({"versions": ["default", "v2"], "default": "v2"})
        self.assertEqual(mgi.merge_visualizers(existing, ["v3"]), ["default", "v2", "v3"])

    def test_object_format_missing_versions_key(self):
        existing = json.dumps({"default": "v2"})
        self.assertEqual(mgi.merge_visualizers(existing, ["v2"]), ["v2"])

    def test_union_dedupes_and_sorts(self):
        existing = json.dumps(["v2", "default"])
        self.assertEqual(mgi.merge_visualizers(existing, ["v2", "default"]), ["default", "v2"])


class BuildIndexTest(unittest.TestCase):
    def test_no_default(self):
        self.assertEqual(mgi.build_index(["default", "v2"]), {"versions": ["default", "v2"]})

    def test_with_default(self):
        self.assertEqual(
            mgi.build_index(["default", "v2"], "v2"),
            {"versions": ["default", "v2"], "default": "v2"},
        )

    def test_default_not_in_versions_is_omitted(self):
        # A misconfigured default must never point at a missing visualizer.
        self.assertEqual(
            mgi.build_index(["default"], "v2"),
            {"versions": ["default"]},
        )

    def test_empty_default_is_omitted(self):
        self.assertEqual(mgi.build_index(["default"], ""), {"versions": ["default"]})


class LoadDefaultVisualizersTest(unittest.TestCase):
    def test_missing_file_returns_empty(self):
        self.assertEqual(mgi.load_default_visualizers("/nonexistent/path.json"), {})

    def test_none_path_returns_empty(self):
        self.assertEqual(mgi.load_default_visualizers(None), {})

    def test_reads_mapping(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump({"open_spiel_chess": "v2"}, f)
            path = f.name
        try:
            self.assertEqual(
                mgi.load_default_visualizers(path), {"open_spiel_chess": "v2"}
            )
        finally:
            os.unlink(path)


class ShippedConfigTest(unittest.TestCase):
    def test_chess_default_is_v2(self):
        config_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "config", "default-visualizers.json"
        )
        defaults = mgi.load_default_visualizers(config_path)
        self.assertEqual(defaults.get("open_spiel_chess"), "v2")


if __name__ == "__main__":
    unittest.main()
