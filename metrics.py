"""OCR evaluation metric helpers.

This module contains functions for computing raw and normalized OCR
error metrics, including character error rate (CER) and word error rate
(WER). It supports raw Levenshtein-based calculations and normalizes
text with jiwer.
"""

import Levenshtein
from jiwer import cer as jiwer_cer, wer as jiwer_wer


def calculate_lev_dist_text(reference: str, predicted: str) -> float:
    """Calculates the Character Error Rate (CER) between a reference text and a predicted text."""

    # Calculates CER using the Levenshtein distance
    lev_dist_text = Levenshtein.distance(predicted, reference)

    return lev_dist_text


def calculate_lev_dist_words(reference: str, predicted: str) -> tuple[float, int]:
    """Calculates raw WER using whitespace tokenization and word-level Levenshtein distance."""

    # Splits the predicted and reference texts into lists of strings (words)
    # This is a strict raw WER regime: no lowercasing, no punctuation normalization,
    # and no additional tokenization beyond Python whitespace splitting.
    predicted_words = predicted.split()
    reference_words = reference.split()

    # Calculates WER by computing Levenshtein distance between two lists of words.
    # If two words don't match exactly, that counts as one word substitution.
    lev_dist_words = Levenshtein.distance(predicted_words, reference_words)

    return lev_dist_words, len(reference_words)


def calculate_jiwer_metrics(reference: str, predicted: str) -> tuple[float, float]:
    """Calculates CER and WER using jiwer default normalization transforms.

    jiwer normalizes both reference and hypothesis before measuring errors.
    The default transform includes lowercasing, punctuation removal, and whitespace
    normalization. This regime is designed to be more robust for OCR evaluation
    than raw whitespace tokenization.
    """
    # Return percentages for consistency with other helpers
    return jiwer_cer(reference, predicted) * 100, jiwer_wer(reference, predicted) * 100


def calculate_jiwer_document_level(
    references: list[str], predictions: list[str]
) -> tuple[float, float]:
    """Calculate document-level jiwer CER and WER.

    The function joins the list of per-page references and predictions with
    newlines and computes jiwer metrics on the concatenated texts. This is
    preferable for a single overall normalized score because jiwer's text
    transforms (lowercasing, punctuation removal, whitespace normalization)
    are applied consistently across the whole document.
    Returns percentages (CER%, WER%).
    """
    reference_all = "\n".join(references)
    predicted_all = "\n".join(predictions)

    cer_percent = jiwer_cer(reference_all, predicted_all) * 100
    wer_percent = jiwer_wer(reference_all, predicted_all) * 100

    return cer_percent, wer_percent
