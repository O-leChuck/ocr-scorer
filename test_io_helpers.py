"""
Unit tests for I/O helper functions.

This module verifies the folder selection and validation behavior,
particularly the validate_and_select_folders function which ensures
matching file counts between GT and prediction folders.
"""

import os
import tempfile
import unittest
from unittest.mock import patch
from io_helpers import validate_and_select_folders


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
                return self.gt_dir if len(call_args_list) == 1 else self.pred_dir
            else:
                # Second attempt: use fixed directories
                return gt_dir_fixed if len(call_args_list) == 3 else pred_dir_fixed

        mock_select.side_effect = track_calls

        result = validate_and_select_folders(
            self.temp_path, self.temp_path, "Select GT", "Select Pred"
        )

        self.assertIsNotNone(result)
        # On retry, the initial_dir should be the last selected folder
        # (not the original temp_path)
        self.assertEqual(call_args_list[2], self.gt_dir)  # Retry uses gt_dir as default
        self.assertEqual(
            call_args_list[3], self.pred_dir
        )  # Retry uses pred_dir as default

    @patch("io_helpers.select_folder")
    def test_only_txt_files_are_counted(self, mock_select):
        """Test that only .txt files are counted, not other files."""
        self._create_text_files(self.gt_dir, 3)
        self._create_text_files(self.pred_dir, 3)

        # Add non-.txt files that shouldn't be counted
        with open(os.path.join(self.gt_dir, "readme.md"), "w", encoding="utf-8") as f:
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


if __name__ == "__main__":
    unittest.main()
