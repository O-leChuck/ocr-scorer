"""
Unit tests for pure I/O helper functions (no GUI/tkinter involved).

Covers metric/log export, the inf/NaN JSON-safety conversion, and the
page-number extraction/alignment heuristics. The interactive folder
dialogs (which do use tkinter) are tested separately in
test_dialogs.py.
"""

import json
import os
import tempfile
import unittest

from ocr_scorer.io_helpers import (
    _make_json_safe,
    check_page_number_alignment,
    extract_page_number,
    format_page_number_check_report,
    make_evaluation_output_folder,
    save_document_metrics,
    save_evaluation_log,
    save_metrics,
)


class TestSaveMetrics(unittest.TestCase):
    """Unit tests for page-wise CSV/JSON export."""

    def setUp(self):
        """Create a temporary output directory for the export files."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_dir = self.temp_dir.name

    def tearDown(self):
        """Clean up the temporary output directory."""
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

    def test_infinite_and_nan_values_round_trip_through_csv_and_json(self):
        """
        Test that an empty-reference page (infinite raw CER/WER) and a
        both-empty page (undefined raw CER/WER) are represented as real
        numbers in the CSV and as explicit, unmistakable values in JSON.
        """
        page_metrics = [
            {
                "page": "hallucinated",
                "cer_raw": float("inf"),
                "wer_raw": float("inf"),
                "cer_jiwer_normalized": 500.0,
                "wer_jiwer_normalized": 100.0,
            },
            {
                "page": "both-empty",
                "cer_raw": float("nan"),
                "wer_raw": float("nan"),
                "cer_jiwer_normalized": 0.0,
                "wer_jiwer_normalized": 0.0,
            },
        ]

        save_metrics(page_metrics, self.output_dir)

        csv_path = os.path.join(self.output_dir, "metrics_pagewise.csv")
        with open(csv_path, encoding="utf-8") as f:
            csv_text = f.read()
        self.assertIn(",inf,inf,", csv_text)
        # NaN is written as an empty CSV field, not a fabricated number
        self.assertIn("both-empty,,,", csv_text)

        json_path = os.path.join(self.output_dir, "metrics_pagewise.json")
        with open(json_path, encoding="utf-8") as f:
            records = json.load(f)
        self.assertEqual(records[0]["cer_raw"], "Infinity")
        self.assertEqual(records[0]["wer_raw"], "Infinity")
        self.assertIsNone(records[1]["cer_raw"])
        self.assertIsNone(records[1]["wer_raw"])

    def test_verbose_false_suppresses_all_prints(self):
        """Test that verbose=False produces no stdout output at all."""
        import contextlib
        import io

        page_metrics = [
            {
                "page": "p1",
                "cer_raw": 0.0,
                "wer_raw": 0.0,
                "cer_jiwer_normalized": 0.0,
                "wer_jiwer_normalized": 0.0,
            }
        ]
        document_metrics = {
            "summary": {"cer_raw": 0.0, "page_count": 1},
            "top_error_chars_raw": [],
            "top_error_chars_normalized": [],
        }

        captured = io.StringIO()
        with contextlib.redirect_stdout(captured):
            save_metrics(
                page_metrics,
                self.output_dir,
                document_metrics=document_metrics,
                verbose=False,
            )

        self.assertEqual(captured.getvalue(), "")


class TestMakeJsonSafe(unittest.TestCase):
    """Unit tests for the inf/NaN -> JSON-safe value conversion."""

    def test_positive_infinity_becomes_string(self):
        """
        Test that positive infinity is converted to the string "Infinity".
        """
        self.assertEqual(_make_json_safe(float("inf")), "Infinity")

    def test_negative_infinity_becomes_string(self):
        """
        Test that negative infinity is converted to the string "-Infinity".
        """
        self.assertEqual(_make_json_safe(float("-inf")), "-Infinity")

    def test_nan_becomes_none(self):
        """
        Test that NaN is converted to None.
        """
        self.assertIsNone(_make_json_safe(float("nan")))

    def test_ordinary_float_is_unchanged(self):
        """
        Test that ordinary finite floats are returned unchanged.
        """
        self.assertEqual(_make_json_safe(5.5), 5.5)

    def test_recurses_into_nested_dicts_and_lists(self):
        """
        Test that the function correctly recurses into nested dictionaries and
        lists, converting any inf/NaN values it finds.
        """
        value = {
            "summary": {"cer_raw": float("inf"), "wer_raw": 2.0},
            "items": [{"count": float("nan")}, {"count": 3}],
        }
        result = _make_json_safe(value)
        self.assertEqual(result["summary"]["cer_raw"], "Infinity")
        self.assertEqual(result["summary"]["wer_raw"], 2.0)
        self.assertIsNone(result["items"][0]["count"])
        self.assertEqual(result["items"][1]["count"], 3)


class TestSaveDocumentMetrics(unittest.TestCase):
    """Unit tests for document-level metric export."""

    def setUp(self):
        """Create a temporary output directory for the export files."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_dir = self.temp_dir.name

    def tearDown(self):
        """Clean up the temporary output directory."""
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

    def test_infinite_summary_value_becomes_infinity_string_in_json(self):
        """
        Test that a document-wide summary value of +inf (the fully
        degenerate all-empty-reference case) is written as the string
        "Infinity" in the JSON, not null and not a fabricated number.
        """
        document_metrics = {
            "summary": {
                "cer_raw": float("inf"),
                "wer_raw": 0.0,
                "page_count": 1,
            },
            "top_error_chars_raw": [],
            "top_error_chars_normalized": [],
        }

        save_document_metrics(document_metrics, self.output_dir)

        json_path = os.path.join(self.output_dir, "metrics_document.json")
        with open(json_path, encoding="utf-8") as f:
            loaded = json.load(f)
        self.assertEqual(loaded["summary"]["cer_raw"], "Infinity")

        summary_csv = os.path.join(
            self.output_dir, "metrics_document_summary.csv"
        )
        with open(summary_csv, encoding="utf-8") as f:
            csv_text = f.read()
        self.assertIn("cer_raw,inf", csv_text)


class TestMakeEvaluationOutputFolder(unittest.TestCase):
    """Unit tests for the dated evaluation output folder helper."""

    def setUp(self):
        """Create a temporary parent directory for the output folder."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = self.temp_dir.name

    def tearDown(self):
        """Clean up the temporary parent directory."""
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
        """Create temporary GT/prediction/output directories."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = self.temp_dir.name
        self.gt_dir = os.path.join(self.temp_path, "gt")
        self.pred_dir = os.path.join(self.temp_path, "pred")
        self.output_dir = os.path.join(self.temp_path, "out")
        os.makedirs(self.gt_dir)
        os.makedirs(self.pred_dir)
        os.makedirs(self.output_dir)

    def tearDown(self):
        """Clean up the temporary GT/prediction/output directories."""
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
        """Test that the page-number check report is written to the log."""
        open(os.path.join(self.gt_dir, "a.txt"), "w", encoding="utf-8").close()
        open(
            os.path.join(self.pred_dir, "a.txt"), "w", encoding="utf-8"
        ).close()

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
        """Test that a "p"-prefixed, zero-padded number is extracted."""
        self.assertEqual(
            extract_page_number(
                "p0004_Miller-T_Auswanderung-Wuerttemberg_EVAL_SAMPLE-20.txt"
            ),
            4,
        )

    def test_p_prefix_at_end_of_filename(self):
        """
        Test that the heuristic recognizes a "p" prefix at the end of the
        filename, even if there are other digits earlier in the name.
        """
        self.assertEqual(
            extract_page_number(
                "TYP_GER_GHII-270_Bericht-GSTHO_EVAL-SAMPLE-50" "_p0142.txt"
            ),
            142,
        )

    def test_page_word_prefix(self):
        """
        Test that the heuristic recognizes "page" as a prefix for the page
        number.
        """
        self.assertEqual(extract_page_number("scan_page7.txt"), 7)

    def test_lone_digit_run_with_no_marker(self):
        """Test that a filename's only digit run is used as a fallback."""
        self.assertEqual(extract_page_number("0004.txt"), 4)

    def test_common_gt_pred_naming_matches(self):
        """Test the project's own gt_file-XX/pred_file-XX convention."""
        self.assertEqual(extract_page_number("gt_file-01.txt"), 1)
        self.assertEqual(extract_page_number("pred_file-01.txt"), 1)

    def test_no_digits_returns_none(self):
        """Test that a filename with no digits yields no page number."""
        self.assertIsNone(
            extract_page_number("some_random_ocr_output_name.txt")
        )

    def test_multiple_p_markers_is_ambiguous(self):
        """Test that two "p"-marked numbers in one filename are ambiguous."""
        self.assertIsNone(extract_page_number("p01_page02.txt"))

    def test_multiple_digit_runs_with_no_marker_is_ambiguous(self):
        """Test that unmarked, multiple digit runs are left unresolved."""
        self.assertIsNone(extract_page_number("scan_2024_v3.txt"))


class TestCheckPageNumberAlignment(unittest.TestCase):
    """Unit tests for the sort-order pairing sanity check."""

    def test_matching_page_numbers_report_no_mismatches(self):
        """Test that consistently numbered files report no mismatches."""
        used, messages = check_page_number_alignment(
            ["gt_file-01.txt", "gt_file-02.txt"],
            ["pred_file-01.txt", "pred_file-02.txt"],
        )
        self.assertTrue(used)
        self.assertEqual(messages, [])

    def test_mismatched_page_numbers_are_reported(self):
        """Test that a page-number mismatch at a given position is
        reported with both filenames and page numbers."""
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
        """Test that unresolvable filenames skip the check without
        reporting any (potentially false) mismatches."""
        used, messages = check_page_number_alignment(
            ["some_random_name.txt"], ["another_random_name.txt"]
        )
        self.assertFalse(used)
        self.assertEqual(messages, [])

    def test_one_unextractable_filename_skips_entire_check(self):
        """Test that even if every other file is extractable, a single
        ambiguous filename disables the check rather than partially
        applying it."""
        used, messages = check_page_number_alignment(
            ["p0001_a.txt", "scan_2024_v3.txt"],
            ["p0001_x.txt", "p0002_y.txt"],
        )
        self.assertFalse(used)
        self.assertEqual(messages, [])


class TestFormatPageNumberCheckReport(unittest.TestCase):
    """Unit tests for the human-readable rendering of the check result."""

    def test_not_used_reports_skip(self):
        """Test that a skipped check is reported as such, in one line."""
        lines = format_page_number_check_report(False, [])
        self.assertEqual(len(lines), 1)
        self.assertIn("skipped", lines[0])

    def test_used_with_no_mismatches_reports_success(self):
        """Test that a clean check reports success, in one line."""
        lines = format_page_number_check_report(True, [])
        self.assertEqual(len(lines), 1)
        self.assertIn("correctly paired", lines[0])

    def test_used_with_mismatches_lists_each_one(self):
        """Test that every reported mismatch appears in its own line."""
        lines = format_page_number_check_report(
            True, ["Position 1: mismatch A", "Position 3: mismatch B"]
        )
        self.assertIn("2 potential mismatch(es)", lines[0])
        self.assertTrue(any("mismatch A" in line for line in lines))
        self.assertTrue(any("mismatch B" in line for line in lines))


if __name__ == "__main__":
    unittest.main()
