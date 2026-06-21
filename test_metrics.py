"""
Unit tests for metrics normalization functions.

This module verifies normalization behavior used by the OCR scorer's
WER pipeline, especially the punctuation-preserving transform.
"""

import unittest

from jiwer import (
    Compose,
    ToLowerCase,
    Strip,
    RemoveMultipleSpaces,
    ReduceToListOfListOfWords,
    wer,
)

from metrics import ReplacePunctuationWithSpace


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


if __name__ == "__main__":
    unittest.main()
