"""
End-to-end tests for main.py.

These drive the full pipeline (folder validation is mocked to avoid
tkinter dialogs; find_folder's directory walk is mocked to keep tests
fast and independent of the host filesystem) and check both the happy
path and the two crash scenarios that were fixed:
  - both folders contain zero .txt files
  - every file that is found fails to open

Expected raw CER/WER percentages for the DS-1/DS-2 fixtures are derived
from distances that were independently cross-validated with a
from-scratch DP implementation (see test_metrics.py), not copied from
test-data/target-results.md, which is documented as incomplete/unverified.
"""

import contextlib
import glob
import io
import json
import os
import tempfile
import unittest
from unittest.mock import patch

import matplotlib

import main

matplotlib.use("Agg")

TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "test-data")


def _run_main_with_folders(gt_dir, pred_dir):
    with (
        patch("main.find_folder", return_value=None),
        patch(
            "main.validate_and_select_folders", return_value=(gt_dir, pred_dir)
        ),
    ):
        main.main()


class TestMainDefaultPathPriority(unittest.TestCase):
    """Tests for main.py's config.ini > auto-detect > fallback priority
    for the folder-dialog starting directory."""

    def _resolve(self, configured, auto_detected):
        """Run main() with the given configured/auto-detected paths and
        return the (initial_directory_gt, initial_directory_pred) that
        were actually passed to validate_and_select_folders."""
        with (
            patch("main.load_default_paths", return_value=configured),
            patch("main.find_folder", side_effect=auto_detected),
            patch(
                "main.validate_and_select_folders", return_value=None
            ) as mock_validate,
        ):
            main.main()
        args, _ = mock_validate.call_args
        return args[0], args[1]

    def test_configured_paths_take_priority(self):
        """Test that config.ini paths win over auto-detected ones."""
        gt, pred = self._resolve(
            configured=("/configured/gt", "/configured/pred"),
            auto_detected=["/auto/gt", "/auto/pred"],
        )
        self.assertEqual(gt, "/configured/gt")
        self.assertEqual(pred, "/configured/pred")

    def test_auto_detected_used_when_not_configured(self):
        """Test that find_folder's result is used when config.ini is
        absent or has no value for that path."""
        gt, pred = self._resolve(
            configured=(None, None),
            auto_detected=["/auto/gt", "/auto/pred"],
        )
        self.assertEqual(gt, "/auto/gt")
        self.assertEqual(pred, "/auto/pred")

    def test_builtin_fallback_used_when_nothing_else_available(self):
        """Test that the hardcoded default is the last resort."""
        gt, pred = self._resolve(
            configured=(None, None), auto_detected=[None, None]
        )
        self.assertEqual(
            gt, "/home/covid10/Nextcloud/Lumen-Lucernae/sources"
        )
        self.assertEqual(
            pred, "/home/covid10/Nextcloud/Lumen-Lucernae/predictions/"
        )

    def test_mixed_sources_are_resolved_independently(self):
        """Test that GT and prediction paths fall through the priority
        chain independently of one another."""
        gt, pred = self._resolve(
            configured=("/configured/gt", None),
            auto_detected=[None, "/auto/pred"],
        )
        self.assertEqual(gt, "/configured/gt")
        self.assertEqual(pred, "/auto/pred")


class TestMainEndToEndFixtures(unittest.TestCase):
    """Runs the real pipeline against the curated test-data/ fixtures."""

    def setUp(self):
        if not os.path.isdir(TEST_DATA_DIR):
            self.skipTest("test-data/ not present")

    def _latest_output_dir(self, pred_dir):
        parent = os.path.dirname(pred_dir)
        candidates = [
            os.path.join(parent, name)
            for name in os.listdir(parent)
            if name.startswith("evaluation_")
        ]
        self.assertEqual(len(candidates), 1)
        return candidates[0]

    def test_ds1_no_errors_yields_zero_percent_everywhere(self):
        """
        Test that test evaluation of dataset 1 yields correct CER/WER
        percentages
        """
        gt_dir = os.path.join(TEST_DATA_DIR, "DS-1_no-errors", "gt")
        pred_dir = os.path.join(TEST_DATA_DIR, "DS-1_no-errors", "pred")

        _run_main_with_folders(gt_dir, pred_dir)

        output_dir = self._latest_output_dir(pred_dir)
        try:
            with open(
                os.path.join(output_dir, "metrics_document.json"),
                encoding="utf-8",
            ) as f:
                doc = json.load(f)

            summary = doc["summary"]
            self.assertEqual(summary["page_count"], 3)
            self.assertEqual(summary["cer_raw"], 0.0)
            self.assertEqual(summary["wer_raw"], 0.0)
            self.assertEqual(summary["cer_normalized"], 0.0)
            self.assertEqual(summary["wer_normalized"], 0.0)
        finally:
            _cleanup(output_dir)

    def test_ds2_regular_errors_matches_cross_validated_percentages(self):
        """
        Test that evaluation of test dataset 2 yields the expected CER/WER
        percentages
        """

        gt_dir = os.path.join(TEST_DATA_DIR, "DS-2_regular-errors", "gt")
        pred_dir = os.path.join(TEST_DATA_DIR, "DS-2_regular-errors", "pred")

        _run_main_with_folders(gt_dir, pred_dir)

        output_dir = self._latest_output_dir(pred_dir)
        try:
            with open(
                os.path.join(output_dir, "metrics_document.json"),
                encoding="utf-8",
            ) as f:
                doc = json.load(f)

            summary = doc["summary"]
            self.assertEqual(summary["page_count"], 3)
            # char distances [4, 6, 4] over ref chars [153, 407, 127]
            expected_cer_raw = (4 + 6 + 4) / (153 + 407 + 127) * 100
            # word distances [3, 6, 4] over ref words [24, 71, 24]
            expected_wer_raw = (3 + 6 + 4) / (24 + 71 + 24) * 100
            self.assertAlmostEqual(
                summary["cer_raw"], expected_cer_raw, places=4
            )
            self.assertAlmostEqual(
                summary["wer_raw"], expected_wer_raw, places=4
            )

            csv_path = os.path.join(output_dir, "metrics_pagewise.csv")
            png_path = os.path.join(output_dir, "metrics_visualization.png")
            pdf_path = os.path.join(output_dir, "evaluation_report.pdf")
            log_path = os.path.join(output_dir, "evaluation_log.txt")
            for path in (csv_path, png_path, pdf_path, log_path):
                self.assertTrue(os.path.isfile(path), f"missing: {path}")
        finally:
            _cleanup(output_dir)

    def test_ds5_all_empty_references_yields_infinite_document_wide_cer(self):
        """
        Test the fully degenerate case: every page has an empty
        reference but the OCR still produced text (hallucination), so
        the document-wide raw CER/WER must be reported as infinite
        rather than the old, misleadingly finite fallback value.
        """
        gt_dir = os.path.join(TEST_DATA_DIR, "DS-5_empty-gt-files", "gt")
        pred_dir = os.path.join(TEST_DATA_DIR, "DS-5_empty-gt-files", "pred")

        _run_main_with_folders(gt_dir, pred_dir)

        output_dir = self._latest_output_dir(pred_dir)
        try:
            with open(
                os.path.join(output_dir, "metrics_document.json"),
                encoding="utf-8",
            ) as f:
                doc = json.load(f)
            self.assertEqual(doc["summary"]["cer_raw"], "Infinity")
            self.assertEqual(doc["summary"]["wer_raw"], "Infinity")

            with open(
                os.path.join(output_dir, "metrics_pagewise.json"),
                encoding="utf-8",
            ) as f:
                pages = json.load(f)
            self.assertEqual(len(pages), 3)
            for page in pages:
                self.assertEqual(page["cer_raw"], "Infinity")
                self.assertEqual(page["wer_raw"], "Infinity")

            # the chart/PDF must still be generated, not crash
            for filename in ("metrics_visualization.png", "evaluation_report.pdf"):
                self.assertTrue(
                    os.path.isfile(os.path.join(output_dir, filename))
                )
        finally:
            _cleanup(output_dir)

    def test_ds6_empty_predictions_are_unaffected_finite_100_percent(self):
        """
        Test that an empty *prediction* against a real reference is
        unaffected by the empty-reference fix: this is a normal,
        finite division (distance == reference length), not a
        divide-by-zero, so it should remain a plain 100% CER/WER.
        """
        gt_dir = os.path.join(TEST_DATA_DIR, "DS-6_empty-pred-files", "gt")
        pred_dir = os.path.join(TEST_DATA_DIR, "DS-6_empty-pred-files", "pred")

        _run_main_with_folders(gt_dir, pred_dir)

        output_dir = self._latest_output_dir(pred_dir)
        try:
            with open(
                os.path.join(output_dir, "metrics_document.json"),
                encoding="utf-8",
            ) as f:
                doc = json.load(f)
            self.assertEqual(doc["summary"]["cer_raw"], 100.0)
            self.assertEqual(doc["summary"]["wer_raw"], 100.0)
        finally:
            _cleanup(output_dir)


class TestMainBothEmptyPage(unittest.TestCase):
    """Regression test for the true 0/0 case (empty ref AND empty pred)."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = self.temp_dir.name
        self.gt_dir = os.path.join(self.temp_path, "gt")
        self.pred_dir = os.path.join(self.temp_path, "pred")
        os.makedirs(self.gt_dir)
        os.makedirs(self.pred_dir)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_both_empty_page_is_not_a_number_not_infinity_not_zero(self):
        open(os.path.join(self.gt_dir, "p0001.txt"), "w", encoding="utf-8").close()
        open(
            os.path.join(self.pred_dir, "p0001.txt"), "w", encoding="utf-8"
        ).close()

        _run_main_with_folders(self.gt_dir, self.pred_dir)

        output_dirs = [
            os.path.join(self.temp_path, name)
            for name in os.listdir(self.temp_path)
            if name.startswith("evaluation_")
        ]
        self.assertEqual(len(output_dirs), 1)
        output_dir = output_dirs[0]

        with open(
            os.path.join(output_dir, "metrics_pagewise.json"), encoding="utf-8"
        ) as f:
            pages = json.load(f)
        self.assertEqual(len(pages), 1)
        self.assertIsNone(pages[0]["cer_raw"])
        self.assertIsNone(pages[0]["wer_raw"])

        with open(
            os.path.join(output_dir, "metrics_document.json"), encoding="utf-8"
        ) as f:
            doc = json.load(f)
        self.assertIsNone(doc["summary"]["cer_raw"])
        self.assertIsNone(doc["summary"]["wer_raw"])


class TestMainEmptyFolders(unittest.TestCase):
    """Regression tests for the crash fixed when there is nothing to score."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = self.temp_dir.name
        self.gt_dir = os.path.join(self.temp_path, "gt")
        self.pred_dir = os.path.join(self.temp_path, "pred")
        os.makedirs(self.gt_dir)
        os.makedirs(self.pred_dir)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_zero_txt_files_exits_without_crashing(self):
        """
        Test that program does not crash when both folders (predictions and
        Goldstandard) exist but contain no .txt files.
        """

        # both folders exist but contain no .txt files at all; an
        # uncaught exception here would already fail the test on its own,
        # with a full traceback, so no try/except is needed
        _run_main_with_folders(self.gt_dir, self.pred_dir)

        # no evaluation_* output folder should have been created, since
        # main() now bails out before creating one
        created = [
            name
            for name in os.listdir(self.temp_path)
            if name.startswith("evaluation_")
        ]
        self.assertEqual(created, [])

    def test_all_files_unreadable_exits_without_crashing(self):
        """
        Test that program does not crash when all .txt files found cannot be
        opened (e.g., due to permission errors).
        """

        # Files that glob("*.txt") matches but that cannot be opened as
        # text (here: a directory named "page.txt") deterministically
        # reproduce "every read fails" regardless of user privileges.
        os.makedirs(os.path.join(self.gt_dir, "page.txt"))
        os.makedirs(os.path.join(self.pred_dir, "page.txt"))

        _run_main_with_folders(self.gt_dir, self.pred_dir)


class TestMainPageNumberCheck(unittest.TestCase):
    """Regression tests for the page-number alignment sanity check."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = self.temp_dir.name
        self.gt_dir = os.path.join(self.temp_path, "gt")
        self.pred_dir = os.path.join(self.temp_path, "pred")
        os.makedirs(self.gt_dir)
        os.makedirs(self.pred_dir)

    def tearDown(self):
        self.temp_dir.cleanup()

    def _run_and_capture(self):
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            _run_main_with_folders(self.gt_dir, self.pred_dir)
        return stdout.getvalue()

    def _read_log(self):
        output_dirs = [
            os.path.join(self.temp_path, name)
            for name in os.listdir(self.temp_path)
            if name.startswith("evaluation_")
        ]
        self.assertEqual(len(output_dirs), 1)
        with open(
            os.path.join(output_dirs[0], "evaluation_log.txt"),
            encoding="utf-8",
        ) as f:
            return f.read()

    def test_matching_page_numbers_are_reported_as_correctly_paired(self):
        """
        Test that when the page numbers in the two folders match, the terminal
        output and the log file both report "correctly paired" rather than
        "potential mismatch".
        """
        with open(
            os.path.join(self.gt_dir, "p0001_a.txt"), "w", encoding="utf-8"
        ) as f:
            f.write("hello world")
        with open(
            os.path.join(self.pred_dir, "p0001_b.txt"), "w", encoding="utf-8"
        ) as f:
            f.write("hello world")

        stdout_text = self._run_and_capture()
        log_text = self._read_log()

        self.assertIn("correctly paired", stdout_text)
        self.assertIn("correctly paired", log_text)

    def test_mismatched_page_numbers_are_reported_in_terminal_and_log(self):
        """
        Test that when the page numbers in the two folders do not match, the
        terminal output and the log file both report "potential mismatch" and
        include the page numbers in the message.
        """
        with open(
            os.path.join(self.gt_dir, "p0001_a.txt"), "w", encoding="utf-8"
        ) as f:
            f.write("hello world")
        with open(
            os.path.join(self.pred_dir, "p0099_b.txt"), "w", encoding="utf-8"
        ) as f:
            f.write("hello world")

        stdout_text = self._run_and_capture()
        log_text = self._read_log()

        for text in (stdout_text, log_text):
            self.assertIn("potential mismatch", text)
            self.assertIn("page 1", text)
            self.assertIn("page 99", text)

    def test_unnumbered_filenames_report_check_was_skipped(self):
        """
        Test that files without numbered filenames are reported as skipped.
        """

        with open(
            os.path.join(self.gt_dir, "some_random_name.txt"),
            "w",
            encoding="utf-8",
        ) as f:
            f.write("hello world")
        with open(
            os.path.join(self.pred_dir, "another_random_name.txt"),
            "w",
            encoding="utf-8",
        ) as f:
            f.write("hello world")

        stdout_text = self._run_and_capture()
        log_text = self._read_log()

        self.assertIn("was skipped", stdout_text)
        self.assertIn("was skipped", log_text)


def _cleanup(output_dir):
    for path in glob.glob(os.path.join(output_dir, "*")):
        os.remove(path)
    os.rmdir(output_dir)


if __name__ == "__main__":
    unittest.main()
