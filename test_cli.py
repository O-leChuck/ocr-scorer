"""
Tests for ocr_scorer.cli, the interactive/command-line entry point.

Folder-selection dialogs (ocr_scorer.dialogs) are mocked throughout -
this module never opens a real tkinter window. One end-to-end test at
the bottom exercises the real wiring between cli.main() and
evaluate.run_evaluation() against a curated fixture, without mocking
run_evaluation itself, as a smoke test that the two modules are
connected correctly.
"""

import glob
import json
import os
import unittest
from pathlib import Path
from unittest.mock import patch

import ocr_scorer.cli as cli

TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "test-data")


class TestDefaultPathPriority(unittest.TestCase):
    """Tests for cli.py's config.ini > fallback priority for the
    folder-dialog starting directory."""

    def _resolve(self, configured):
        """Run main() with the given configured paths and return the
        (initial_directory_gt, initial_directory_pred) that were
        actually passed to validate_and_select_folders."""
        with (
            patch(
                "ocr_scorer.cli.load_default_paths", return_value=configured
            ),
            patch(
                "ocr_scorer.cli.validate_and_select_folders", return_value=None
            ) as mock_validate,
        ):
            cli.main([])
        args, _ = mock_validate.call_args
        return args[0], args[1]

    def test_configured_paths_take_priority(self):
        """Test that config.ini paths win over the built-in fallback."""
        gt, pred = self._resolve(("/configured/gt", "/configured/pred"))
        self.assertEqual(gt, "/configured/gt")
        self.assertEqual(pred, "/configured/pred")

    def test_home_directory_used_when_not_configured(self):
        """Test that the user's home directory (portable across
        Windows/macOS/Linux, and free of any assumption about this
        machine's folder structure) is used when config.ini is absent
        or has no value for that path."""
        gt, pred = self._resolve((None, None))
        self.assertEqual(gt, str(Path.home()))
        self.assertEqual(pred, str(Path.home()))

    def test_mixed_sources_are_resolved_independently(self):
        """Test that GT and prediction paths fall through the priority
        chain independently of one another."""
        gt, pred = self._resolve(("/configured/gt", None))
        self.assertEqual(gt, "/configured/gt")
        self.assertEqual(pred, str(Path.home()))


class TestCliArgs(unittest.TestCase):
    """Tests for main()'s --gt/--pred CLI flags, for shell-level
    pipeline use that bypasses the interactive folder dialogs."""

    def test_both_flags_given_skip_dialogs_and_config(self):
        """Test that --gt/--pred are used directly, without touching
        config.ini or showing any folder-selection dialog."""
        with (
            patch("ocr_scorer.cli.run_evaluation") as mock_run,
            patch("ocr_scorer.cli.load_default_paths") as mock_config,
            patch("ocr_scorer.cli.validate_and_select_folders") as mock_dialog,
        ):
            cli.main(["--gt", "/some/gt", "--pred", "/some/pred"])

        mock_run.assert_called_once_with("/some/gt", "/some/pred")
        mock_config.assert_not_called()
        mock_dialog.assert_not_called()

    def test_only_one_flag_given_is_an_error(self):
        """Test that providing only one of --gt/--pred doesn't silently
        fall back to the interactive dialog with a half-set value."""
        with (
            patch("ocr_scorer.cli.run_evaluation") as mock_run,
            patch("ocr_scorer.cli.validate_and_select_folders") as mock_dialog,
        ):
            cli.main(["--gt", "/some/gt"])

        mock_run.assert_not_called()
        mock_dialog.assert_not_called()

    def test_value_error_from_run_evaluation_is_reported_not_raised(self):
        """Test that main() catches run_evaluation's ValueError and
        reports it, rather than letting a human see a raw traceback."""
        with patch(
            "ocr_scorer.cli.run_evaluation",
            side_effect=ValueError("file count mismatch"),
        ):
            try:
                cli.main(["--gt", "/some/gt", "--pred", "/some/pred"])
            except ValueError:
                self.fail("main() should catch ValueError, not propagate it")


class TestMainInteractiveFlow(unittest.TestCase):
    """Tests for the interactive (no CLI flags) folder-dialog flow."""

    def test_user_cancelling_the_dialog_exits_without_calling_run_evaluation(
        self,
    ):
        """Test that a cancelled dialog (validate_and_select_folders
        returning None) exits cleanly without attempting to evaluate."""
        with (
            patch(
                "ocr_scorer.cli.load_default_paths", return_value=(None, None)
            ),
            patch(
                "ocr_scorer.cli.validate_and_select_folders",
                return_value=None,
            ),
            patch("ocr_scorer.cli.run_evaluation") as mock_run,
        ):
            cli.main([])

        mock_run.assert_not_called()

    def test_selected_folders_are_passed_through_to_run_evaluation(self):
        """Test that the folders returned by the dialog are exactly
        what gets passed to run_evaluation."""
        with (
            patch(
                "ocr_scorer.cli.load_default_paths", return_value=(None, None)
            ),
            patch(
                "ocr_scorer.cli.validate_and_select_folders",
                return_value=("/picked/gt", "/picked/pred"),
            ),
            patch("ocr_scorer.cli.run_evaluation") as mock_run,
        ):
            cli.main([])

        mock_run.assert_called_once_with("/picked/gt", "/picked/pred")


class TestCliEndToEnd(unittest.TestCase):
    """One real, unmocked run through cli.main() -> run_evaluation(),
    as a smoke test that the two modules are wired together correctly.
    The more detailed evaluation-logic coverage lives in
    test_evaluate.py; this is deliberately not duplicating that."""

    def setUp(self):
        """Skip if the curated test-data/ fixtures aren't present."""
        if not os.path.isdir(TEST_DATA_DIR):
            self.skipTest("test-data/ not present")

    def test_gt_pred_flags_produce_real_output_files(self):
        """Test that --gt/--pred flags actually run a full evaluation
        end-to-end and write the expected output files."""
        gt_dir = os.path.join(TEST_DATA_DIR, "DS-1_no-errors", "gt")
        pred_dir = os.path.join(TEST_DATA_DIR, "DS-1_no-errors", "pred")

        cli.main(["--gt", gt_dir, "--pred", pred_dir])

        parent = os.path.dirname(pred_dir)
        candidates = [
            os.path.join(parent, name)
            for name in os.listdir(parent)
            if name.startswith("evaluation_")
        ]
        self.assertEqual(len(candidates), 1)
        output_dir = candidates[0]
        try:
            with open(
                os.path.join(output_dir, "metrics_document.json"),
                encoding="utf-8",
            ) as f:
                doc = json.load(f)
            self.assertEqual(doc["summary"]["page_count"], 3)
        finally:
            for path in glob.glob(os.path.join(output_dir, "*")):
                os.remove(path)
            os.rmdir(output_dir)


if __name__ == "__main__":
    unittest.main()
