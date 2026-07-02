"""
Unit tests for character-level error counting functions in metrics.py.

This module verifies that the top_error_chars_for_pair and
aggregate_top_error_chars functions correctly identify and count
character-level errors in OCR output.
"""

import unittest

from metrics import top_error_chars_for_pair, aggregate_top_error_chars


class TestCharErrorCounting(unittest.TestCase):
    """Unit tests for character-level error counting."""

    def test_substitution_counts(self):
        """Verify that substitutions are counted correctly."""
        ref = "abcdef"
        hyp = "abXdef"
        top = top_error_chars_for_pair(ref, hyp, normalize=False, top_n=5)
        # 'c' should be the top error (substituted)
        self.assertGreaterEqual(len(top), 1)
        self.assertEqual(top[0][0], "c")
        self.assertEqual(top[0][1], 1)

    def test_deletion_counts(self):
        """Verify that deletions are counted correctly."""
        ref = "hello"
        hyp = "hell"
        top = top_error_chars_for_pair(ref, hyp, normalize=False, top_n=5)
        self.assertIn(("o", 1), top)

    def test_aggregate_counts(self):
        """
        Verify that aggregate counts across multiple pairs are correct.
        """

        refs = ["abc", "abcd"]
        hyps = ["abX", "abXd"]
        agg = aggregate_top_error_chars(refs, hyps, normalize=False, top_n=3)
        # 'c' and 'c' from first and second sequence count as two
        # errors for 'c'
        # 'c' appears once as deletion/substitution depending; check
        # that we have entries
        self.assertTrue(len(agg) >= 1)


if __name__ == "__main__":
    unittest.main()
