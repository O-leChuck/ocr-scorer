"""
Unit tests for config.py.

These verify that load_default_paths() falls back to (None, None) in
every situation where a configured path shouldn't be trusted (missing
file, missing section/key, empty value, or a path that no longer
exists on disk), and only returns a path when it is genuinely usable.
"""

import os
import tempfile
import unittest
from unittest.mock import patch

import ocr_scorer.config as config


def _write_config(path, contents):
    with open(path, "w", encoding="utf-8") as f:
        f.write(contents)


class TestConfigPathLocation(unittest.TestCase):
    """Regression test for a real bug: _CONFIG_PATH used to be computed
    as os.path.dirname(config.py's own file), which was correct when
    config.py lived at the project root, but silently broke when it
    moved into the ocr_scorer/ package (one directory deeper) - it then
    pointed at the nonexistent ocr_scorer/config.ini instead of the
    real project-root config.ini, so a user's configured paths were
    silently ignored with no error at all."""

    def test_config_path_is_at_project_root_not_inside_the_package(self):
        """Test that _CONFIG_PATH's directory is the parent of the
        ocr_scorer package directory, not the package directory itself."""
        package_dir = os.path.dirname(os.path.abspath(config.__file__))
        project_root = os.path.dirname(package_dir)

        self.assertEqual(os.path.dirname(config._CONFIG_PATH), project_root)
        self.assertNotEqual(
            os.path.dirname(config._CONFIG_PATH), package_dir
        )

    def test_config_path_is_next_to_the_tracked_template_file(self):
        """Test against a real, known anchor: config.template.ini is
        tracked in git at the project root, so _CONFIG_PATH's directory
        must contain it. This would have caught the original bug
        directly, since the buggy path pointed at a directory that
        doesn't contain config.template.ini at all."""
        template_path = os.path.join(
            os.path.dirname(config._CONFIG_PATH), "config.template.ini"
        )
        self.assertTrue(
            os.path.isfile(template_path),
            f"config.template.ini not found next to _CONFIG_PATH "
            f"({template_path}) - _CONFIG_PATH may be pointing at the "
            "wrong directory",
        )


class TestLoadDefaultPaths(unittest.TestCase):
    """Unit tests for config.load_default_paths()."""

    def setUp(self):
        """Create a temporary directory to hold gt/pred/config folders."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = self.temp_dir.name
        self.config_path = os.path.join(self.temp_path, "config.ini")
        self.gt_dir = os.path.join(self.temp_path, "gt")
        self.pred_dir = os.path.join(self.temp_path, "pred")
        os.makedirs(self.gt_dir)
        os.makedirs(self.pred_dir)

    def tearDown(self):
        """Clean up the temporary directory."""
        self.temp_dir.cleanup()

    def _load_with_patched_path(self):
        with patch.object(config, "_CONFIG_PATH", self.config_path):
            return config.load_default_paths()

    def test_missing_config_file_returns_none_none(self):
        """Test that a nonexistent config.ini yields (None, None)."""
        self.assertEqual(self._load_with_patched_path(), (None, None))

    def test_valid_paths_are_returned(self):
        """Test that valid, existing paths are read correctly."""
        _write_config(
            self.config_path,
            f"[paths]\n"
            f"goldstandard_folder = {self.gt_dir}\n"
            f"prediction_folder = {self.pred_dir}\n",
        )

        gt, pred = self._load_with_patched_path()

        self.assertEqual(gt, self.gt_dir)
        self.assertEqual(pred, self.pred_dir)

    def test_empty_values_return_none(self):
        """Test that blank config values are treated as unset."""
        _write_config(
            self.config_path,
            "[paths]\ngoldstandard_folder = \nprediction_folder = \n",
        )

        self.assertEqual(self._load_with_patched_path(), (None, None))

    def test_missing_section_returns_none_none(self):
        """Test that a config.ini without a [paths] section is safe."""
        _write_config(self.config_path, "[other]\nkey = value\n")

        self.assertEqual(self._load_with_patched_path(), (None, None))

    def test_nonexistent_configured_path_is_ignored(self):
        """Test that a configured path pointing nowhere real is dropped,
        rather than being handed to the rest of the app as valid."""
        missing_path = os.path.join(self.temp_path, "does_not_exist")
        _write_config(
            self.config_path,
            f"[paths]\n"
            f"goldstandard_folder = {missing_path}\n"
            f"prediction_folder = {self.pred_dir}\n",
        )

        gt, pred = self._load_with_patched_path()

        self.assertIsNone(gt)
        self.assertEqual(pred, self.pred_dir)

    def test_malformed_config_file_returns_none_none(self):
        """Test that an unparsable config.ini doesn't raise, just warns
        and falls back to (None, None)."""
        _write_config(self.config_path, "this is not valid ini [[[")

        self.assertEqual(self._load_with_patched_path(), (None, None))

    def test_one_path_configured_other_left_blank(self):
        """Test that only the configured half of the pair is returned,
        rather than an all-or-nothing result."""
        _write_config(
            self.config_path,
            f"[paths]\ngoldstandard_folder = {self.gt_dir}\n",
        )

        gt, pred = self._load_with_patched_path()

        self.assertEqual(gt, self.gt_dir)
        self.assertIsNone(pred)


if __name__ == "__main__":
    unittest.main()
