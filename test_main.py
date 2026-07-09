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

import glob
import json
import os
import tempfile
import unittest
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")

import main

TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "test-data")


def _run_main_with_folders(gt_dir, pred_dir):
    with patch("main.find_folder", return_value=None), patch(
        "main.validate_and_select_folders", return_value=(gt_dir, pred_dir)
    ):
        main.main()


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
        gt_dir = os.path.join(TEST_DATA_DIR, "DS-1_no-errors", "gt")
        pred_dir = os.path.join(TEST_DATA_DIR, "DS-1_no-errors", "pred")

        _run_main_with_folders(gt_dir, pred_dir)

        output_dir = self._latest_output_dir(pred_dir)
        try:
            with open(
                os.path.join(output_dir, "metrics_document.json"), encoding="utf-8"
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
        gt_dir = os.path.join(TEST_DATA_DIR, "DS-2_regular-errors", "gt")
        pred_dir = os.path.join(TEST_DATA_DIR, "DS-2_regular-errors", "pred")

        _run_main_with_folders(gt_dir, pred_dir)

        output_dir = self._latest_output_dir(pred_dir)
        try:
            with open(
                os.path.join(output_dir, "metrics_document.json"), encoding="utf-8"
            ) as f:
                doc = json.load(f)

            summary = doc["summary"]
            self.assertEqual(summary["page_count"], 3)
            # char distances [4, 6, 4] over ref chars [153, 407, 127]
            expected_cer_raw = (4 + 6 + 4) / (153 + 407 + 127) * 100
            # word distances [3, 6, 4] over ref words [24, 71, 24]
            expected_wer_raw = (3 + 6 + 4) / (24 + 71 + 24) * 100
            self.assertAlmostEqual(summary["cer_raw"], expected_cer_raw, places=4)
            self.assertAlmostEqual(summary["wer_raw"], expected_wer_raw, places=4)

            csv_path = os.path.join(output_dir, "metrics_pagewise.csv")
            png_path = os.path.join(output_dir, "metrics_visualization.png")
            pdf_path = os.path.join(output_dir, "evaluation_report.pdf")
            log_path = os.path.join(output_dir, "evaluation_log.txt")
            for path in (csv_path, png_path, pdf_path, log_path):
                self.assertTrue(os.path.isfile(path), f"missing: {path}")
        finally:
            _cleanup(output_dir)


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
        # both folders exist but contain no .txt files at all
        try:
            _run_main_with_folders(self.gt_dir, self.pred_dir)
        except Exception as exc:  # pragma: no cover - failure path
            self.fail(f"main() raised unexpectedly on empty folders: {exc!r}")

        # no evaluation_* output folder should have been created, since
        # main() now bails out before creating one
        created = [
            name for name in os.listdir(self.temp_path) if name.startswith("evaluation_")
        ]
        self.assertEqual(created, [])

    def test_all_files_unreadable_exits_without_crashing(self):
        # Files that glob("*.txt") matches but that cannot be opened as
        # text (here: a directory named "page.txt") deterministically
        # reproduce "every read fails" regardless of user privileges.
        os.makedirs(os.path.join(self.gt_dir, "page.txt"))
        os.makedirs(os.path.join(self.pred_dir, "page.txt"))

        try:
            _run_main_with_folders(self.gt_dir, self.pred_dir)
        except Exception as exc:  # pragma: no cover - failure path
            self.fail(f"main() raised unexpectedly when all reads fail: {exc!r}")


def _cleanup(output_dir):
    for path in glob.glob(os.path.join(output_dir, "*")):
        os.remove(path)
    os.rmdir(output_dir)


if __name__ == "__main__":
    unittest.main()
