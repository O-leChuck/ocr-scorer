"""
Unit tests for metrics.py.

Covers the normalization transforms, the raw and jiwer-normalized CER/WER
calculations, character-error attribution, and cross-checks against the
curated fixtures in test-data/.
"""

import glob
import os
import unittest

from jiwer import (
    Compose,
    ToLowerCase,
    Strip,
    RemoveMultipleSpaces,
    ReduceToListOfListOfWords,
    wer,
)

from metrics import (
    ReplacePunctuationWithSpace,
    aggregate_top_error_chars,
    calculate_jiwer_counts,
    calculate_jiwer_metrics,
    calculate_lev_dist_text,
    calculate_lev_dist_words,
    top_error_chars_for_pair,
)

TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "test-data")


class TestMetricsNormalization(unittest.TestCase):
    """Unit tests for normalization behavior in metrics.py."""

    def test_replace_punctuation_with_space_preserves_word_boundaries(self):
        """Verify that punctuation is converted to spaces for normalized WER."""
        wer_transform = Compose(
            [
                ToLowerCase(),
                ReplacePunctuationWithSpace(),
                RemoveMultipleSpaces(),
                Strip(),
                ReduceToListOfListOfWords(),
            ]
        )

        self.assertEqual(
            wer(
                "high-quality",
                "high quality",
                reference_transform=wer_transform,
                hypothesis_transform=wer_transform,
            ),
            0.0,
        )
        self.assertEqual(
            wer(
                "foo,bar",
                "foo bar",
                reference_transform=wer_transform,
                hypothesis_transform=wer_transform,
            ),
            0.0,
        )
        self.assertEqual(
            wer(
                "Mr.Smith",
                "Mr Smith",
                reference_transform=wer_transform,
                hypothesis_transform=wer_transform,
            ),
            0.0,
        )


class TestCalculateLevDistText(unittest.TestCase):
    """Unit tests for the raw, unnormalized character-level distance."""

    def test_identical_strings(self):
        self.assertEqual(calculate_lev_dist_text("abc", "abc"), 0)

    def test_single_substitution(self):
        self.assertEqual(calculate_lev_dist_text("abc", "abd"), 1)

    def test_case_sensitivity(self):
        # raw regime: no lowercasing, "A" != "a"
        self.assertEqual(calculate_lev_dist_text("abc", "Abc"), 1)

    def test_empty_reference_returns_length_of_prediction(self):
        self.assertEqual(calculate_lev_dist_text("", "abc"), 3)

    def test_empty_prediction_returns_length_of_reference(self):
        self.assertEqual(calculate_lev_dist_text("abc", ""), 3)

    def test_both_empty_is_zero(self):
        self.assertEqual(calculate_lev_dist_text("", ""), 0)


class TestCalculateLevDistWords(unittest.TestCase):
    """Unit tests for the raw, whitespace-tokenized word-level distance."""

    def test_identical_returns_zero_and_word_count(self):
        dist, ref_words = calculate_lev_dist_words("a b c", "a b c")
        self.assertEqual(dist, 0)
        self.assertEqual(ref_words, 3)

    def test_single_word_substitution(self):
        dist, ref_words = calculate_lev_dist_words("a b c", "a x c")
        self.assertEqual(dist, 1)
        self.assertEqual(ref_words, 3)

    def test_empty_reference_returns_prediction_word_count(self):
        dist, ref_words = calculate_lev_dist_words("", "a b c")
        self.assertEqual(dist, 3)
        self.assertEqual(ref_words, 0)

    def test_empty_prediction_returns_reference_word_count(self):
        dist, ref_words = calculate_lev_dist_words("a b c", "")
        self.assertEqual(dist, 3)
        self.assertEqual(ref_words, 3)

    def test_both_empty(self):
        dist, ref_words = calculate_lev_dist_words("", "")
        self.assertEqual(dist, 0)
        self.assertEqual(ref_words, 0)

    def test_extra_whitespace_does_not_create_phantom_words(self):
        # raw split() collapses any amount of whitespace, unlike naive
        # split(" ") which would produce empty-string "words"
        dist, ref_words = calculate_lev_dist_words("a  b   c", "a b c")
        self.assertEqual(dist, 0)
        self.assertEqual(ref_words, 3)


class TestCalculateJiwerMetrics(unittest.TestCase):
    """Unit tests for the normalized CER/WER regime, documented in the README."""

    def test_case_difference_is_normalized_away(self):
        cer, wer_pct = calculate_jiwer_metrics("Hello World", "hello world")
        self.assertEqual(cer, 0.0)
        self.assertEqual(wer_pct, 0.0)

    def test_punctuation_difference_is_normalized_away(self):
        cer, wer_pct = calculate_jiwer_metrics("Hello, World!", "Hello World")
        self.assertEqual(cer, 0.0)
        self.assertEqual(wer_pct, 0.0)

    def test_leading_and_trailing_whitespace_is_stripped(self):
        cer, wer_pct = calculate_jiwer_metrics(" word ", "word")
        self.assertEqual(cer, 0.0)
        self.assertEqual(wer_pct, 0.0)

    def test_multiple_spaces_are_collapsed_for_wer(self):
        _, wer_pct = calculate_jiwer_metrics("word1  word2", "word1 word2")
        self.assertEqual(wer_pct, 0.0)

    def test_both_empty_is_zero(self):
        cer, wer_pct = calculate_jiwer_metrics("", "")
        self.assertEqual(cer, 0.0)
        self.assertEqual(wer_pct, 0.0)

    def test_counts_are_consistent_with_percentage_on_normal_text(self):
        # calculate_jiwer_counts must report the same edit/reference
        # totals that calculate_jiwer_metrics' percentage is derived from.
        reference, predicted = "Hello, World!", "hello world unexpected"
        cer_pct, wer_pct = calculate_jiwer_metrics(reference, predicted)
        char_edits, ref_chars, word_edits, ref_words = calculate_jiwer_counts(
            reference, predicted
        )
        self.assertAlmostEqual(cer_pct, (char_edits / ref_chars) * 100)
        self.assertAlmostEqual(wer_pct, (word_edits / ref_words) * 100)


class TestTopErrorCharsForPair(unittest.TestCase):
    """Unit tests for per-pair character error attribution."""

    def test_substitution_is_attributed_to_reference_char(self):
        self.assertEqual(
            top_error_chars_for_pair("cat", "cbt", normalize=False), [("a", 1)]
        )

    def test_deletion_is_attributed_to_reference_char(self):
        self.assertEqual(
            top_error_chars_for_pair("cats", "cat", normalize=False), [("s", 1)]
        )

    def test_insertion_is_not_attributed_to_any_char(self):
        # "cat" -> "cats" is a pure insertion; no reference char was
        # misrecognized, so nothing should be counted.
        self.assertEqual(top_error_chars_for_pair("cat", "cats", normalize=False), [])

    def test_non_alphabetic_errors_are_excluded(self):
        self.assertEqual(top_error_chars_for_pair("a1b", "a2b", normalize=False), [])

    def test_identical_strings_have_no_errors(self):
        self.assertEqual(top_error_chars_for_pair("abc", "abc", normalize=False), [])


class TestAggregateTopErrorChars(unittest.TestCase):
    """Unit tests for corpus-wide character error aggregation."""

    def test_aggregates_and_ranks_across_pairs(self):
        results = aggregate_top_error_chars(
            ["cat", "dog"], ["cbt", "dpg"], normalize=False, top_n=5
        )
        self.assertEqual(
            sorted(results), sorted([("a", 1, 50.0), ("o", 1, 50.0)])
        )

    def test_no_errors_returns_empty_list(self):
        self.assertEqual(
            aggregate_top_error_chars(["cat"], ["cat"], normalize=False, top_n=5), []
        )

    def test_top_n_truncates_results(self):
        references = ["aaaa", "bbb", "cc"]
        predictions = ["xxxx", "yyy", "zz"]
        results = aggregate_top_error_chars(
            references, predictions, normalize=False, top_n=2
        )
        self.assertEqual(len(results), 2)
        # 'a' has the most errors (4), so it must be ranked first
        self.assertEqual(results[0][0], "a")


class TestFixtureDataRawDistances(unittest.TestCase):
    """Regression tests against test-data/, using raw char/word distances
    that were independently cross-validated with a from-scratch DP
    implementation (see project history) rather than trusted from
    test-data/target-results.md, which is documented as incomplete/unverified.

    Only the raw Levenshtein-based quantities are asserted here, since
    they are unambiguous. Percentage aggregation with empty references
    (DS-5/DS-6) is intentionally not asserted; that behavior is under
    review (see project discussion on empty-reference handling).
    """

    # (dataset, filename) -> (char_dist, ref_chars, word_dist, ref_words)
    EXPECTED = {
        ("DS-1_no-errors", "gt_file-01.txt"): (0, 153, 0, 24),
        ("DS-1_no-errors", "gt_file-02.txt"): (0, 407, 0, 71),
        ("DS-1_no-errors", "gt_file-03.txt"): (0, 127, 0, 24),
        ("DS-2_regular-errors", "gt_file-01.txt"): (4, 153, 3, 24),
        ("DS-2_regular-errors", "gt_file-02.txt"): (6, 407, 6, 71),
        ("DS-2_regular-errors", "gt_file-03.txt"): (4, 127, 4, 24),
        ("DS-3_case-errors", "gt_file-01.txt"): (7, 153, 5, 24),
        ("DS-4_edge-cases", "gt_file-01.txt"): (56, 407, 10, 71),
        ("DS-4_edge-cases", "gt_file-02.txt"): (407, 407, 71, 71),
        ("DS-4_edge-cases", "gt_file-03.txt"): (127, 0, 24, 0),
        ("DS-4_edge-cases", "gt_file-04.txt"): (0, 0, 0, 0),
        ("DS-4_edge-cases", "gt_file-05.txt"): (30, 153, 5, 24),
        ("DS-4_edge-cases", "gt_file-06.txt"): (5, 153, 0, 24),
        ("DS-4_edge-cases", "gt_file-07.txt"): (67, 407, 11, 71),
        ("DS-5_empty-gt-files", "gt_file-01.txt"): (153, 0, 24, 0),
        ("DS-5_empty-gt-files", "gt_file-02.txt"): (407, 0, 71, 0),
        ("DS-5_empty-gt-files", "gt_file-03.txt"): (127, 0, 24, 0),
        ("DS-6_empty-pred-files", "gt_file-01.txt"): (153, 153, 24, 24),
        ("DS-6_empty-pred-files", "gt_file-02.txt"): (407, 407, 71, 71),
        ("DS-6_empty-pred-files", "gt_file-03.txt"): (127, 127, 24, 24),
    }

    def test_fixture_raw_distances_match_cross_validated_values(self):
        if not os.path.isdir(TEST_DATA_DIR):
            self.skipTest("test-data/ not present")

        for (dataset, gt_name), (
            expected_char_dist,
            expected_ref_chars,
            expected_word_dist,
            expected_ref_words,
        ) in self.EXPECTED.items():
            gt_path = os.path.join(TEST_DATA_DIR, dataset, "gt", gt_name)
            pred_dir = os.path.join(TEST_DATA_DIR, dataset, "pred")
            # prediction files share the same basename in DS-1..DS-3/DS-5,
            # but DS-6 predictions are named "pred_file-*.txt"
            pred_name = gt_name.replace("gt_file", "pred_file")
            pred_candidates = [
                os.path.join(pred_dir, gt_name),
                os.path.join(pred_dir, pred_name),
            ]
            pred_path = next((p for p in pred_candidates if os.path.isfile(p)), None)

            with self.subTest(dataset=dataset, file=gt_name):
                if pred_path is None:
                    self.fail("prediction file not found")
                with open(gt_path, encoding="utf-8") as f:
                    reference = f.read()
                with open(pred_path, encoding="utf-8") as f:
                    predicted = f.read()

                self.assertEqual(len(reference), expected_ref_chars)
                self.assertEqual(
                    calculate_lev_dist_text(reference, predicted),
                    expected_char_dist,
                )

                word_dist, ref_words = calculate_lev_dist_words(reference, predicted)
                self.assertEqual(ref_words, expected_ref_words)
                self.assertEqual(word_dist, expected_word_dist)

    def test_ds3_edge_cases_folders_are_empty(self):
        # DS-3_edge-cases exercises the "zero files in either folder"
        # case used to reproduce the empty-folder crash fix in main.py.
        gt_dir = os.path.join(TEST_DATA_DIR, "DS-3_edge-cases", "gt")
        pred_dir = os.path.join(TEST_DATA_DIR, "DS-3_edge-cases", "pred")
        if not os.path.isdir(gt_dir):
            self.skipTest("test-data/DS-3_edge-cases not present")
        self.assertEqual(glob.glob(os.path.join(gt_dir, "*.txt")), [])
        self.assertEqual(glob.glob(os.path.join(pred_dir, "*.txt")), [])


if __name__ == "__main__":
    unittest.main()
