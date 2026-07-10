"""
Unit tests for I/O helper functions.

This module verifies the folder selection and validation behavior,
particularly the validate_and_select_folders function which ensures
matching file counts between GT and prediction folders, plus the
metric/log export helpers.
"""

import json
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from io_helpers import (
    check_page_number_alignment,
    extract_page_number,
    find_folder,
    format_page_number_check_report,
    make_evaluation_output_folder,
    save_document_metrics,
    save_evaluation_log,
    save_metrics,
    select_folder,
    validate_and_select_folders,
)


class TestValidateAndSelectFolders(unittest.TestCase):
    """Unit tests for folder validation and selection."""

    def setUp(self):
        """Create temporary directories with test files for each test."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = self.temp_dir.name

        # Create subdirectories for GT and predictions
        self.gt_dir = os.path.join(self.temp_path, "gt")
        self.pred_dir = os.path.join(self.temp_path, "pred")
        os.makedirs(self.gt_dir)
        os.makedirs(self.pred_dir)

    def tearDown(self):
        """Clean up temporary directories."""
        self.temp_dir.cleanup()

    def _create_text_files(self, directory: str, count: int) -> None:
        """Helper to create count text files in directory."""
        for i in range(1, count + 1):
            filepath = os.path.join(directory, f"file-{i:02d}.txt")
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(f"Sample content {i}")

    @patch("io_helpers.select_folder")
    def test_matching_file_counts_returns_folders(self, mock_select):
        """Test that matching file counts return the folder tuple."""
        self._create_text_files(self.gt_dir, 3)
        self._create_text_files(self.pred_dir, 3)

        # Mock select_folder to return our test directories
        mock_select.side_effect = [self.gt_dir, self.pred_dir]

        result = validate_and_select_folders(
            self.temp_path, self.temp_path, "Select GT", "Select Pred"
        )

        self.assertIsNotNone(result)
        self.assertEqual(result, (self.gt_dir, self.pred_dir))
        # Should only be called twice (once for GT, once for pred)
        self.assertEqual(mock_select.call_count, 2)

    @patch("io_helpers.showerror")
    @patch("io_helpers.select_folder")
    def test_mismatching_file_counts_shows_error_and_retries(
        self, mock_select, mock_showerror
    ):
        """Test that mismatch shows error dialog and retries."""
        self._create_text_files(self.gt_dir, 3)
        self._create_text_files(self.pred_dir, 5)

        gt_dir_fixed = os.path.join(self.temp_path, "gt_fixed")
        pred_dir_fixed = os.path.join(self.temp_path, "pred_fixed")
        os.makedirs(gt_dir_fixed)
        os.makedirs(pred_dir_fixed)
        self._create_text_files(gt_dir_fixed, 4)
        self._create_text_files(pred_dir_fixed, 4)

        # First attempt: mismatch. Second attempt: match.
        mock_select.side_effect = [
            self.gt_dir,
            self.pred_dir,
            gt_dir_fixed,
            pred_dir_fixed,
        ]

        result = validate_and_select_folders(
            self.temp_path, self.temp_path, "Select GT", "Select Pred"
        )

        self.assertIsNotNone(result)
        self.assertEqual(result, (gt_dir_fixed, pred_dir_fixed))
        # select_folder should be called 4 times (2 per attempt, 2 attempts)
        self.assertEqual(mock_select.call_count, 4)
        # Error dialog should be shown once
        mock_showerror.assert_called_once()
        # Check that error message contains file counts
        error_call = mock_showerror.call_args
        error_message = error_call[0][1]
        self.assertIn("3 files", error_message)
        self.assertIn("5 files", error_message)

    @patch("io_helpers.select_folder")
    def test_user_cancels_on_gt_selection(self, mock_select):
        """Test that returning None (user cancels) exits gracefully."""
        # User cancels on first dialog
        mock_select.return_value = ""

        result = validate_and_select_folders(
            self.temp_path, self.temp_path, "Select GT", "Select Pred"
        )

        self.assertIsNone(result)
        # Should only call once (user cancelled on first dialog)
        mock_select.assert_called_once()

    @patch("io_helpers.select_folder")
    def test_user_cancels_on_pred_selection(self, mock_select):
        """Test user cancelling on prediction folder selection."""
        self._create_text_files(self.gt_dir, 3)

        # User selects GT, then cancels on pred selection
        mock_select.side_effect = [self.gt_dir, ""]

        result = validate_and_select_folders(
            self.temp_path, self.temp_path, "Select GT", "Select Pred"
        )

        self.assertIsNone(result)
        # Should call twice (GT selected, then user cancels on pred)
        self.assertEqual(mock_select.call_count, 2)

    @patch("io_helpers.select_folder")
    def test_empty_folders_match(self, mock_select):
        """Test that two empty folders are considered matching."""
        # Both directories are empty (no .txt files)
        mock_select.side_effect = [self.gt_dir, self.pred_dir]

        result = validate_and_select_folders(
            self.temp_path, self.temp_path, "Select GT", "Select Pred"
        )

        self.assertIsNotNone(result)
        self.assertEqual(result, (self.gt_dir, self.pred_dir))

    @patch("io_helpers.select_folder")
    def test_sticky_defaults_on_retry(self, mock_select):
        """Test that last selected folders become defaults on retry."""
        self._create_text_files(self.gt_dir, 3)
        self._create_text_files(self.pred_dir, 5)

        gt_dir_fixed = os.path.join(self.temp_path, "gt_fixed")
        pred_dir_fixed = os.path.join(self.temp_path, "pred_fixed")
        os.makedirs(gt_dir_fixed)
        os.makedirs(pred_dir_fixed)
        self._create_text_files(gt_dir_fixed, 4)
        self._create_text_files(pred_dir_fixed, 4)

        # Mock select_folder to track initial_dir arguments
        call_args_list = []

        def track_calls(initial_dir, _title):
            call_args_list.append(initial_dir)
            if len(call_args_list) <= 2:
                # First attempt: use mismatched directories
                return (
                    self.gt_dir if len(call_args_list) == 1 else self.pred_dir
                )
            else:
                # Second attempt: use fixed directories
                return (
                    gt_dir_fixed
                    if len(call_args_list) == 3
                    else pred_dir_fixed
                )

        mock_select.side_effect = track_calls

        result = validate_and_select_folders(
            self.temp_path, self.temp_path, "Select GT", "Select Pred"
        )

        self.assertIsNotNone(result)
        # On retry, the initial_dir should be the last selected folder
        # (not the original temp_path)
        # Retry uses gt_dir as default
        self.assertEqual(call_args_list[2], self.gt_dir)
        self.assertEqual(
            call_args_list[3], self.pred_dir
        )  # Retry uses pred_dir as default

    @patch("io_helpers.select_folder")
    def test_only_txt_files_are_counted(self, mock_select):
        """Test that only .txt files are counted, not other files."""
        self._create_text_files(self.gt_dir, 3)
        self._create_text_files(self.pred_dir, 3)

        # Add non-.txt files that shouldn't be counted
        with open(
            os.path.join(self.gt_dir, "readme.md"), "w", encoding="utf-8"
        ) as f:
            f.write("readme")
        with open(
            os.path.join(self.pred_dir, "metadata.json"), "w", encoding="utf-8"
        ) as f:
            f.write("{}")

        mock_select.side_effect = [self.gt_dir, self.pred_dir]

        result = validate_and_select_folders(
            self.temp_path, self.temp_path, "Select GT", "Select Pred"
        )

        # Should still match because only .txt files are counted
        self.assertIsNotNone(result)
        self.assertEqual(result, (self.gt_dir, self.pred_dir))


class TestSelectFolder(unittest.TestCase):
    """Unit tests for select_folder's Tk lifecycle management."""

    @patch("io_helpers.askdirectory")
    @patch("io_helpers.Tk")
    def test_tk_root_is_withdrawn_and_destroyed(
        self, mock_tk_cls, mock_askdirectory
    ):
        """
        Test that the Tk root window is withdrawn and destroyed properly.
        """

        mock_root = MagicMock()
        mock_tk_cls.return_value = mock_root
        mock_askdirectory.return_value = "/some/folder"

        result = select_folder("/initial", "Select a folder")

        self.assertEqual(result, "/some/folder")
        mock_root.withdraw.assert_called_once()
        mock_root.destroy.assert_called_once()

    @patch("io_helpers.askdirectory")
    @patch("io_helpers.Tk")
    def test_tk_root_is_destroyed_even_if_dialog_raises(
        self, mock_tk_cls, mock_askdirectory
    ):
        """
        Test that the Tk root window is destroyed even if askdirectory raises
        an exception.
        """
        mock_root = MagicMock()
        mock_tk_cls.return_value = mock_root
        mock_askdirectory.side_effect = RuntimeError("dialog failed")

        with self.assertRaises(RuntimeError):
            select_folder("/initial", "Select a folder")

        mock_root.destroy.assert_called_once()


class TestFindFolder(unittest.TestCase):
    """Unit tests for locating a named folder under a search root."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = self.temp_dir.name

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_finds_folder_in_nested_directory(self):
        """
        Test that find_folder locates a folder nested under the start path.
        """
        target = os.path.join(self.temp_path, "a", "b", "Goldstandard")
        os.makedirs(target)

        found = find_folder("Goldstandard", start_path=self.temp_path)

        self.assertEqual(found, target)

    def test_returns_none_when_not_found(self):
        """
        Test that find_folder returns None when the folder does not exist.
        """
        os.makedirs(os.path.join(self.temp_path, "unrelated"))

        found = find_folder("Goldstandard", start_path=self.temp_path)

        self.assertIsNone(found)


class TestSaveMetrics(unittest.TestCase):
    """Unit tests for page-wise CSV/JSON export."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_dir = self.temp_dir.name

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_writes_csv_and_json_with_expected_rows(self):
        """
        Test that save_metrics writes both CSV and JSON files with the correct
        number of rows.
        """

        page_metrics = [
            {
                "page": "page-01",
                "cer_raw": 0.0,
                "wer_raw": 0.0,
                "cer_jiwer_normalized": 0.0,
                "wer_jiwer_normalized": 0.0,
            },
            {
                "page": "page-02",
                "cer_raw": 5.5,
                "wer_raw": 10.0,
                "cer_jiwer_normalized": 4.0,
                "wer_jiwer_normalized": 8.0,
            },
        ]

        df = save_metrics(page_metrics, self.output_dir)

        self.assertEqual(len(df), 2)
        csv_path = os.path.join(self.output_dir, "metrics_pagewise.csv")
        json_path = os.path.join(self.output_dir, "metrics_pagewise.json")
        self.assertTrue(os.path.isfile(csv_path))
        self.assertTrue(os.path.isfile(json_path))

        with open(json_path, encoding="utf-8") as f:
            records = json.load(f)
        self.assertEqual(len(records), 2)
        self.assertEqual(records[1]["page"], "page-02")
        self.assertEqual(records[1]["cer_raw"], 5.5)

    def test_document_metrics_are_saved_when_provided(self):
        """
        Test to ensure that document-level metrics are saved to JSON and CSV
         files when provided.
        """
        document_metrics = {
            "summary": {"cer_raw": 1.23, "page_count": 1},
            "top_error_chars_raw": [
                {"rank": 1, "character": "a", "count": 3, "percent": 100.0}
            ],
            "top_error_chars_normalized": [],
        }

        save_metrics(
            [
                {
                    "page": "p1",
                    "cer_raw": 0,
                    "wer_raw": 0,
                    "cer_jiwer_normalized": 0,
                    "wer_jiwer_normalized": 0,
                }
            ],
            self.output_dir,
            document_metrics=document_metrics,
        )

        self.assertTrue(
            os.path.isfile(
                os.path.join(self.output_dir, "metrics_document.json")
            )
        )
        self.assertTrue(
            os.path.isfile(
                os.path.join(self.output_dir, "metrics_document_summary.csv")
            )
        )
        self.assertTrue(
            os.path.isfile(
                os.path.join(self.output_dir, "metrics_document_top_raw.csv")
            )
        )
        # empty sections are skipped rather than writing an empty file
        self.assertFalse(
            os.path.isfile(
                os.path.join(
                    self.output_dir, "metrics_document_top_normalized.csv"
                )
            )
        )


class TestSaveDocumentMetrics(unittest.TestCase):
    """Unit tests for document-level metric export."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_dir = self.temp_dir.name

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_json_summary_round_trips(self):
        """
        Test that document metrics are saved to JSON and can be loaded back
         correctly.
        """
        document_metrics = {
            "summary": {"cer_raw": 2.5, "wer_raw": 4.0, "page_count": 3},
            "top_error_chars_raw": [
                {"rank": 1, "character": "e", "count": 10, "percent": 50.0}
            ],
            "top_error_chars_normalized": [
                {"rank": 1, "character": "e", "count": 8, "percent": 40.0}
            ],
        }

        save_document_metrics(document_metrics, self.output_dir)

        json_path = os.path.join(self.output_dir, "metrics_document.json")
        with open(json_path, encoding="utf-8") as f:
            loaded = json.load(f)
        self.assertEqual(loaded, document_metrics)

        summary_csv = os.path.join(
            self.output_dir, "metrics_document_summary.csv"
        )
        top_raw_csv = os.path.join(
            self.output_dir, "metrics_document_top_raw.csv"
        )
        top_norm_csv = os.path.join(
            self.output_dir, "metrics_document_top_normalized.csv"
        )
        self.assertTrue(os.path.isfile(summary_csv))
        self.assertTrue(os.path.isfile(top_raw_csv))
        self.assertTrue(os.path.isfile(top_norm_csv))


class TestMakeEvaluationOutputFolder(unittest.TestCase):
    """Unit tests for the dated evaluation output folder helper."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = self.temp_dir.name

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_creates_dated_folder_next_to_prediction_folder(self):
        """
        Test that the evaluation output folder is created next to the
        predictions folder.
        """
        pred_folder = os.path.join(self.temp_path, "predictions")
        os.makedirs(pred_folder)

        output_dir = make_evaluation_output_folder(pred_folder)

        self.assertTrue(os.path.isdir(output_dir))
        self.assertEqual(os.path.dirname(output_dir), self.temp_path)
        self.assertTrue(os.path.basename(output_dir).startswith("evaluation_"))

    def test_is_idempotent_when_called_twice(self):
        """
        Test that calling make_evaluation_output_folder twice returns the same
        path
        """
        pred_folder = os.path.join(self.temp_path, "predictions")
        os.makedirs(pred_folder)

        first = make_evaluation_output_folder(pred_folder)
        second = make_evaluation_output_folder(pred_folder)

        self.assertEqual(first, second)


class TestSaveEvaluationLog(unittest.TestCase):
    """Unit tests for the evaluation log text export."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = self.temp_dir.name
        self.gt_dir = os.path.join(self.temp_path, "gt")
        self.pred_dir = os.path.join(self.temp_path, "pred")
        self.output_dir = os.path.join(self.temp_path, "out")
        os.makedirs(self.gt_dir)
        os.makedirs(self.pred_dir)
        os.makedirs(self.output_dir)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_log_contains_folder_metadata_and_file_lists(self):
        """
        Test that the evaluation log contains the expected metadata and file
        lists.
        """
        for name in ("a.txt", "b.txt"):
            open(
                os.path.join(self.gt_dir, name), "w", encoding="utf-8"
            ).close()
            open(
                os.path.join(self.pred_dir, name), "w", encoding="utf-8"
            ).close()

        save_evaluation_log(
            self.gt_dir,
            self.pred_dir,
            self.output_dir,
            evaluation_date="2026-01-01",
        )

        log_path = os.path.join(self.output_dir, "evaluation_log.txt")
        self.assertTrue(os.path.isfile(log_path))
        with open(log_path, encoding="utf-8") as f:
            content = f.read()

        self.assertIn("Evaluation date: 2026-01-01", content)
        self.assertIn(self.gt_dir, content)
        self.assertIn(self.pred_dir, content)
        self.assertIn("Page count: 2", content)
        self.assertIn("a.txt", content)
        self.assertIn("b.txt", content)

    def test_log_includes_page_number_check_report(self):
        open(os.path.join(self.gt_dir, "a.txt"), "w", encoding="utf-8").close()
        open(os.path.join(self.pred_dir, "a.txt"), "w", encoding="utf-8").close()

        save_evaluation_log(
            self.gt_dir,
            self.pred_dir,
            self.output_dir,
            evaluation_date="2026-01-01",
            page_number_check=(True, ["Position 1: mismatch example"]),
        )

        log_path = os.path.join(self.output_dir, "evaluation_log.txt")
        with open(log_path, encoding="utf-8") as f:
            content = f.read()

        self.assertIn("Page-number check:", content)
        self.assertIn("Position 1: mismatch example", content)


class TestExtractPageNumber(unittest.TestCase):
    """Unit tests for the page-number extraction heuristic."""

    def test_p_prefix_with_leading_zeros(self):
        self.assertEqual(
            extract_page_number(
                "p0004_Miller-T_Auswanderung-Wuerttemberg_EVAL_SAMPLE-20.txt"
            ),
            4,
        )

    def test_p_prefix_at_end_of_filename(self):
        self.assertEqual(
            extract_page_number(
                "TYP_GER_DSHI-190_Krusenstjern-Bericht-ERR_EVAL-SAMPLE-50_p0142.txt"
            ),
            142,
        )

    def test_page_word_prefix(self):
        self.assertEqual(extract_page_number("scan_page7.txt"), 7)

    def test_lone_digit_run_with_no_marker(self):
        self.assertEqual(extract_page_number("0004.txt"), 4)

    def test_common_gt_pred_naming_matches(self):
        self.assertEqual(extract_page_number("gt_file-01.txt"), 1)
        self.assertEqual(extract_page_number("pred_file-01.txt"), 1)

    def test_no_digits_returns_none(self):
        self.assertIsNone(extract_page_number("some_random_ocr_output_name.txt"))

    def test_multiple_p_markers_is_ambiguous(self):
        self.assertIsNone(extract_page_number("p01_page02.txt"))

    def test_multiple_digit_runs_with_no_marker_is_ambiguous(self):
        self.assertIsNone(extract_page_number("scan_2024_v3.txt"))


class TestCheckPageNumberAlignment(unittest.TestCase):
    """Unit tests for the sort-order pairing sanity check."""

    def test_matching_page_numbers_report_no_mismatches(self):
        used, messages = check_page_number_alignment(
            ["gt_file-01.txt", "gt_file-02.txt"],
            ["pred_file-01.txt", "pred_file-02.txt"],
        )
        self.assertTrue(used)
        self.assertEqual(messages, [])

    def test_mismatched_page_numbers_are_reported(self):
        used, messages = check_page_number_alignment(
            ["p0004_a.txt", "p0005_b.txt"],
            ["weird_output_p0142.txt", "weird_output_p0005.txt"],
        )
        self.assertTrue(used)
        self.assertEqual(len(messages), 1)
        self.assertIn("Position 1", messages[0])
        self.assertIn("page 4", messages[0])
        self.assertIn("page 142", messages[0])

    def test_unextractable_filenames_skip_the_check_silently(self):
        used, messages = check_page_number_alignment(
            ["some_random_name.txt"], ["another_random_name.txt"]
        )
        self.assertFalse(used)
        self.assertEqual(messages, [])

    def test_one_unextractable_filename_skips_entire_check(self):
        # even if every other file is extractable, a single ambiguous
        # filename should disable the check rather than partially apply it
        used, messages = check_page_number_alignment(
            ["p0001_a.txt", "scan_2024_v3.txt"],
            ["p0001_x.txt", "p0002_y.txt"],
        )
        self.assertFalse(used)
        self.assertEqual(messages, [])


class TestFormatPageNumberCheckReport(unittest.TestCase):
    """Unit tests for the human-readable rendering of the check result."""

    def test_not_used_reports_skip(self):
        lines = format_page_number_check_report(False, [])
        self.assertEqual(len(lines), 1)
        self.assertIn("skipped", lines[0])

    def test_used_with_no_mismatches_reports_success(self):
        lines = format_page_number_check_report(True, [])
        self.assertEqual(len(lines), 1)
        self.assertIn("correctly paired", lines[0])

    def test_used_with_mismatches_lists_each_one(self):
        lines = format_page_number_check_report(
            True, ["Position 1: mismatch A", "Position 3: mismatch B"]
        )
        self.assertIn("2 potential mismatch(es)", lines[0])
        self.assertTrue(any("mismatch A" in line for line in lines))
        self.assertTrue(any("mismatch B" in line for line in lines))


if __name__ == "__main__":
    unittest.main()
