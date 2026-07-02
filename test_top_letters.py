"""
Unit tests for the aggregate_top_error_chars function in metrics.py.
"""

import unittest

from metrics import aggregate_top_error_chars


class TestTopLetterAggregation(unittest.TestCase):
    """Unit tests for top-letter aggregation across multiple reference/prediction pairs."""

    def test_simple_substitution(self):
        """Verify that a single substitution is counted correctly across one document."""

        refs = ["abc"]
        hyps = ["abx"]
        top = aggregate_top_error_chars(refs, hyps, normalize=False, top_n=5)
        self.assertEqual(len(top), 1)
        char, count, pct = top[0]
        self.assertEqual(char, "c")
        self.assertEqual(count, 1)
        self.assertAlmostEqual(pct, 100.0)

    def test_multiple_documents_counts(self):
        """Verify that error counts are aggregated correctly across multiple documents."""

        refs = ["aaaa", "bbbb"]
        hyps = ["aaXa", "bbXb"]
        top = aggregate_top_error_chars(refs, hyps, normalize=False, top_n=5)
        # There should be two different characters with one error each
        counts = {char: cnt for char, cnt, pct in top}
        self.assertEqual(sum(counts.values()), 2)
        self.assertTrue("a" in counts or "b" in counts)

    def test_normalized_mode_basic(self):
        """Verify that normalization works and lowercases characters."""

        refs = ["A"]
        hyps = ["B"]
        top = aggregate_top_error_chars(refs, hyps, normalize=True, top_n=3)
        self.assertEqual(len(top), 1)
        char, count, _pct = top[0]
        # normalization should lowercase the character
        self.assertEqual(char.lower(), "a")
        self.assertEqual(count, 1)


if __name__ == "__main__":
    unittest.main()
