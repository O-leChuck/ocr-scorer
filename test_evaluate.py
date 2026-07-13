"""
Tests for ocr_scorer.evaluate.run_evaluation(), the core evaluation
logic and the intended import point for pipeline/programmatic use.

These call run_evaluation() directly with plain folder paths - no
tkinter dialogs are involved anywhere in this module, matching how a
real caller (either the CLI or another script) would use it.

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

from ocr_scorer.evaluate import run_evaluation
from ocr_scorer.metrics import (
    calculate_jiwer_metrics as _real_calculate_jiwer_metrics,
)

TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "test-data")


def _cleanup(output_dir):
    for path in glob.glob(os.path.join(output_dir, "*")):
        os.remove(path)
    os.rmdir(output_dir)


class TestEndToEndFixtures(unittest.TestCase):
    """Runs the real pipeline against the curated test-data/ fixtures."""

    def setUp(self):
        """Skip if the curated test-data/ fixtures aren't present."""
        if not os.path.isdir(TEST_DATA_DIR):
            self.skipTest("test-data/ not present")

    def test_ds1_no_errors_yields_zero_percent_everywhere(self):
        """
        Test that evaluation of dataset 1 yields correct CER/WER
        percentages.
        """
        gt_dir = os.path.join(TEST_DATA_DIR, "DS-1_no-errors", "gt")
        pred_dir = os.path.join(TEST_DATA_DIR, "DS-1_no-errors", "pred")

        output_dir, document_metrics = run_evaluation(
            gt_dir, pred_dir, verbose=False
        )

        try:
            summary = document_metrics["summary"]
            self.assertEqual(summary["page_count"], 3)
            self.assertEqual(summary["cer_raw"], 0.0)
            self.assertEqual(summary["wer_raw"], 0.0)
            self.assertEqual(summary["cer_normalized"], 0.0)
            self.assertEqual(summary["wer_normalized"], 0.0)
            self.assertEqual(document_metrics["warnings"], [])
        finally:
            _cleanup(output_dir)

    def test_ds2_regular_errors_matches_cross_validated_percentages(self):
        """
        Test that evaluation of test dataset 2 yields the expected CER/WER
        percentages.
        """
        gt_dir = os.path.join(TEST_DATA_DIR, "DS-2_regular-errors", "gt")
        pred_dir = os.path.join(TEST_DATA_DIR, "DS-2_regular-errors", "pred")

        output_dir, document_metrics = run_evaluation(
            gt_dir, pred_dir, verbose=False
        )

        try:
            summary = document_metrics["summary"]
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
        rather than the old, misleadingly finite fallback value - and
        that this is also surfaced in document_metrics["warnings"].
        """
        gt_dir = os.path.join(TEST_DATA_DIR, "DS-5_empty-gt-files", "gt")
        pred_dir = os.path.join(TEST_DATA_DIR, "DS-5_empty-gt-files", "pred")

        output_dir, document_metrics = run_evaluation(
            gt_dir, pred_dir, verbose=False
        )

        try:
            summary = document_metrics["summary"]
            self.assertEqual(summary["cer_raw"], float("inf"))
            self.assertEqual(summary["wer_raw"], float("inf"))
            self.assertTrue(
                any(
                    "raw CER is undefined" in w
                    for w in document_metrics["warnings"]
                )
            )

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
            for filename in (
                "metrics_visualization.png",
                "evaluation_report.pdf",
            ):
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
        divide-by-zero, so it should remain a plain 100% CER/WER with
        no warnings.
        """
        gt_dir = os.path.join(TEST_DATA_DIR, "DS-6_empty-pred-files", "gt")
        pred_dir = os.path.join(TEST_DATA_DIR, "DS-6_empty-pred-files", "pred")

        output_dir, document_metrics = run_evaluation(
            gt_dir, pred_dir, verbose=False
        )

        try:
            summary = document_metrics["summary"]
            self.assertEqual(summary["cer_raw"], 100.0)
            self.assertEqual(summary["wer_raw"], 100.0)
            self.assertEqual(document_metrics["warnings"], [])
        finally:
            _cleanup(output_dir)


class TestRunEvaluationReturnValue(unittest.TestCase):
    """Tests for run_evaluation()'s return value and input validation -
    the contract a pipeline caller depends on."""

    def setUp(self):
        """Skip if the curated test-data/ fixtures aren't present."""
        if not os.path.isdir(TEST_DATA_DIR):
            self.skipTest("test-data/ not present")

    def test_returns_output_dir_and_document_metrics(self):
        """Test the happy path returns a usable (output_dir, metrics)."""
        gt_dir = os.path.join(TEST_DATA_DIR, "DS-2_regular-errors", "gt")
        pred_dir = os.path.join(TEST_DATA_DIR, "DS-2_regular-errors", "pred")

        output_dir, document_metrics = run_evaluation(
            gt_dir, pred_dir, verbose=False
        )

        try:
            self.assertTrue(os.path.isdir(output_dir))
            self.assertEqual(document_metrics["summary"]["page_count"], 3)
            self.assertAlmostEqual(
                document_metrics["summary"]["cer_raw"], 2.0378, places=3
            )
        finally:
            _cleanup(output_dir)

    def test_nonexistent_gt_folder_raises_value_error(self):
        """Test that a missing Goldstandard folder raises, not crashes
        with an unhandled exception or silently does nothing."""
        pred_dir = os.path.join(TEST_DATA_DIR, "DS-2_regular-errors", "pred")

        with self.assertRaises(ValueError):
            run_evaluation("/does/not/exist", pred_dir)

    def test_nonexistent_pred_folder_raises_value_error(self):
        """Test that a missing prediction folder raises."""
        gt_dir = os.path.join(TEST_DATA_DIR, "DS-2_regular-errors", "gt")

        with self.assertRaises(ValueError):
            run_evaluation(gt_dir, "/does/not/exist")

    def test_mismatched_file_counts_raise_value_error(self):
        """Test that folders with different .txt counts raise, since
        there is no interactive retry available to a direct caller."""
        gt_dir = os.path.join(TEST_DATA_DIR, "DS-2_regular-errors", "gt")
        pred_dir = os.path.join(TEST_DATA_DIR, "DS-4_edge-cases", "gt")

        with self.assertRaises(ValueError):
            run_evaluation(gt_dir, pred_dir)

    def test_empty_folders_raise_value_error(self):
        """Test that folders with no .txt files raise, rather than the
        old print-and-return behavior (which a caller can't observe)."""
        with (
            tempfile.TemporaryDirectory() as empty_gt,
            tempfile.TemporaryDirectory() as empty_pred,
        ):
            with self.assertRaises(ValueError):
                run_evaluation(empty_gt, empty_pred)

    def test_all_files_unreadable_raises_value_error(self):
        """
        Test that when every file matched by the folder scan fails to
        open, run_evaluation() raises rather than silently returning.
        Files that glob("*.txt") matches but that cannot be opened as
        text (here: a directory named "page.txt") deterministically
        reproduce "every read fails" regardless of user privileges.
        """
        with (
            tempfile.TemporaryDirectory() as gt_dir,
            tempfile.TemporaryDirectory() as pred_dir,
        ):
            os.makedirs(os.path.join(gt_dir, "page.txt"))
            os.makedirs(os.path.join(pred_dir, "page.txt"))

            with self.assertRaises(ValueError):
                run_evaluation(gt_dir, pred_dir, verbose=False)


class TestBothEmptyPage(unittest.TestCase):
    """Regression test for the true 0/0 case (empty ref AND empty pred)."""

    def setUp(self):
        """Create temporary GT/prediction directories, each with one
        empty .txt file."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = self.temp_dir.name
        self.gt_dir = os.path.join(self.temp_path, "gt")
        self.pred_dir = os.path.join(self.temp_path, "pred")
        os.makedirs(self.gt_dir)
        os.makedirs(self.pred_dir)

    def tearDown(self):
        """Clean up the temporary directories."""
        self.temp_dir.cleanup()

    def test_both_empty_page_is_not_a_number_not_infinity_not_zero(self):
        """Test that a page with both an empty reference and an empty
        prediction reports an undefined (not-a-number) rate, not a
        fabricated 0% or +inf."""
        open(
            os.path.join(self.gt_dir, "p0001.txt"), "w", encoding="utf-8"
        ).close()
        open(
            os.path.join(self.pred_dir, "p0001.txt"), "w", encoding="utf-8"
        ).close()

        output_dir, document_metrics = run_evaluation(
            self.gt_dir, self.pred_dir, verbose=False
        )

        try:
            with open(
                os.path.join(output_dir, "metrics_pagewise.json"),
                encoding="utf-8",
            ) as f:
                pages = json.load(f)
            self.assertEqual(len(pages), 1)
            self.assertIsNone(pages[0]["cer_raw"])
            self.assertIsNone(pages[0]["wer_raw"])

            summary = document_metrics["summary"]
            self.assertNotEqual(summary["cer_raw"], summary["cer_raw"])  # NaN
            self.assertNotEqual(summary["wer_raw"], summary["wer_raw"])  # NaN
        finally:
            _cleanup(output_dir)


class TestPageNumberCheck(unittest.TestCase):
    """Regression tests for the page-number alignment sanity check."""

    def setUp(self):
        """Create temporary GT/prediction directories."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = self.temp_dir.name
        self.gt_dir = os.path.join(self.temp_path, "gt")
        self.pred_dir = os.path.join(self.temp_path, "pred")
        os.makedirs(self.gt_dir)
        os.makedirs(self.pred_dir)

    def tearDown(self):
        """Clean up the temporary directories."""
        self.temp_dir.cleanup()

    def _run_and_capture(self):
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            output_dir, document_metrics = run_evaluation(
                self.gt_dir, self.pred_dir
            )
        return stdout.getvalue(), output_dir, document_metrics

    def _read_log(self, output_dir):
        with open(
            os.path.join(output_dir, "evaluation_log.txt"),
            encoding="utf-8",
        ) as f:
            return f.read()

    def test_matching_page_numbers_are_reported_as_correctly_paired(self):
        """
        Test that when the page numbers in the two folders match, the
        terminal output and the log file both report "correctly
        paired" rather than "potential mismatch", and no warning is
        recorded.
        """
        with open(
            os.path.join(self.gt_dir, "p0001_a.txt"), "w", encoding="utf-8"
        ) as f:
            f.write("hello world")
        with open(
            os.path.join(self.pred_dir, "p0001_b.txt"), "w", encoding="utf-8"
        ) as f:
            f.write("hello world")

        stdout_text, output_dir, document_metrics = self._run_and_capture()
        log_text = self._read_log(output_dir)

        try:
            self.assertIn("correctly paired", stdout_text)
            self.assertIn("correctly paired", log_text)
            self.assertEqual(document_metrics["warnings"], [])
        finally:
            _cleanup(output_dir)

    def test_mismatched_page_numbers_are_reported_in_terminal_and_log(self):
        """
        Test that when the page numbers in the two folders do not
        match, the terminal output, the log file, and
        document_metrics["warnings"] all report the mismatch with the
        page numbers involved.
        """
        with open(
            os.path.join(self.gt_dir, "p0001_a.txt"), "w", encoding="utf-8"
        ) as f:
            f.write("hello world")
        with open(
            os.path.join(self.pred_dir, "p0099_b.txt"), "w", encoding="utf-8"
        ) as f:
            f.write("hello world")

        stdout_text, output_dir, document_metrics = self._run_and_capture()
        log_text = self._read_log(output_dir)

        try:
            for text in (stdout_text, log_text):
                self.assertIn("potential mismatch", text)
                self.assertIn("page 1", text)
                self.assertIn("page 99", text)
            self.assertEqual(len(document_metrics["warnings"]), 1)
            self.assertIn("page 1", document_metrics["warnings"][0])
            self.assertIn("page 99", document_metrics["warnings"][0])
        finally:
            _cleanup(output_dir)

    def test_unnumbered_filenames_report_check_was_skipped(self):
        """
        Test that files without numbered filenames are reported as
        skipped, and no (potentially false) warning is recorded.
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

        stdout_text, output_dir, document_metrics = self._run_and_capture()
        log_text = self._read_log(output_dir)

        try:
            self.assertIn("was skipped", stdout_text)
            self.assertIn("was skipped", log_text)
            self.assertEqual(document_metrics["warnings"], [])
        finally:
            _cleanup(output_dir)


class TestPerPageCalculationResilience(unittest.TestCase):
    """Tests that a calculation failure on one page (as opposed to a
    file-read error, which was already handled) is skipped with a
    warning rather than crashing the whole run and losing every
    already-processed page."""

    def setUp(self):
        """Create three GT/prediction page pairs, one of which will be
        made to fail calculation via a mocked metrics function."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = self.temp_dir.name
        self.gt_dir = os.path.join(self.temp_path, "gt")
        self.pred_dir = os.path.join(self.temp_path, "pred")
        os.makedirs(self.gt_dir)
        os.makedirs(self.pred_dir)

        for name, text in (
            ("p0001.txt", "hello world"),
            ("p0002.txt", "BOOM trigger page"),
            ("p0003.txt", "goodbye world"),
        ):
            with open(
                os.path.join(self.gt_dir, name), "w", encoding="utf-8"
            ) as f:
                f.write(text)
            with open(
                os.path.join(self.pred_dir, name), "w", encoding="utf-8"
            ) as f:
                f.write(text)

    def tearDown(self):
        """Clean up the temporary directories."""
        self.temp_dir.cleanup()

    @staticmethod
    def _flaky_calculate_jiwer_metrics(reference, predicted):
        """Behaves exactly like the real function, except it raises for
        the one page deliberately marked to fail."""
        if "BOOM" in reference:
            raise RuntimeError("simulated calculation failure")
        return _real_calculate_jiwer_metrics(reference, predicted)

    def test_bad_page_is_skipped_with_warning_others_still_complete(self):
        """Test that the bad page is skipped (page_count == 2, not 3),
        the other two pages are still fully evaluated, and a warning
        naming the failure is recorded rather than the run crashing."""
        with patch(
            "ocr_scorer.evaluate.calculate_jiwer_metrics",
            side_effect=self._flaky_calculate_jiwer_metrics,
        ):
            output_dir, document_metrics = run_evaluation(
                self.gt_dir, self.pred_dir, verbose=False
            )

        try:
            self.assertEqual(
                document_metrics["summary"]["page_count"], 2
            )
            self.assertTrue(
                any(
                    "p0002.txt" in w and "simulated calculation failure" in w
                    for w in document_metrics["warnings"]
                )
            )

            with open(
                os.path.join(output_dir, "metrics_pagewise.json"),
                encoding="utf-8",
            ) as f:
                pages = json.load(f)
            page_names = {p["page"] for p in pages}
            self.assertEqual(page_names, {"p0001", "p0003"})
        finally:
            _cleanup(output_dir)


if __name__ == "__main__":
    unittest.main()
