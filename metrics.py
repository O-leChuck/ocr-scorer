"""OCR evaluation metric helpers.

This module contains functions for computing raw and normalized OCR
error metrics, including character error rate (CER) and word error rate
(WER). It supports raw Levenshtein-based calculations and normalizes
text with jiwer.
"""

from collections import Counter
from typing import Iterable, List, Tuple
import unicodedata
import Levenshtein
from jiwer import (
    cer as jiwer_cer,
    process_characters,
    process_words,
    wer as jiwer_wer,
    Compose,
    ToLowerCase,
    RemovePunctuation,
    Strip,
    RemoveMultipleSpaces,
    ReduceToListOfListOfChars,
    ReduceToListOfListOfWords,
)
from jiwer.transforms import AbstractTransform


class ReplacePunctuationWithSpace(AbstractTransform):
    """Replace Unicode punctuation with spaces while preserving token boundaries.

    This transform is used only for normalized WER. It preserves word
    boundaries when punctuation separates text, for example
    "high-quality" -> "high quality".
    """

    def process_string(self, s: str) -> str:
        return "".join(
            " " if unicodedata.category(char).startswith("P") else char for char in s
        )


# Custom jiwer transforms with aggressive normalization.
# This pipeline is intentionally applied to normalized CER/WER so the
# script focuses on transcription accuracy rather than formatting.
#
# Current behavior:
# - Lowercase text
# - Remove Unicode punctuation (categories Po, Pd, Ps, Pe, Pi, Pf, Pc)
# - Strip leading/trailing whitespace
# - Replace punctuation with spaces for WER
# - Collapse multiple spaces for WER
cer_custom_transform = Compose(
    [
        ToLowerCase(),
        RemovePunctuation(),
        Strip(),
        ReduceToListOfListOfChars(),
    ]
)

wer_custom_transform = Compose(
    [
        ToLowerCase(),
        ReplacePunctuationWithSpace(),
        RemoveMultipleSpaces(),
        Strip(),
        ReduceToListOfListOfWords(),
    ]
)


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
    """Calculates CER and WER using jiwer with custom aggressive normalization.

    Custom transforms include lowercasing, punctuation removal, and whitespace
    normalization. This regime is designed to focus on transcription accuracy
    rather than formatting or punctuation differences.

    Transforms applied:
    - ToLowerCase()
    - RemovePunctuation() (removes Unicode category P: punctuation)
    - RemoveMultipleSpaces() (for WER only)
    - Strip()
    """
    # Return percentages for consistency with other helpers
    return (
        jiwer_cer(
            reference,
            predicted,
            reference_transform=cer_custom_transform,
            hypothesis_transform=cer_custom_transform,
        )
        * 100,
        jiwer_wer(
            reference,
            predicted,
            reference_transform=wer_custom_transform,
            hypothesis_transform=wer_custom_transform,
        )
        * 100,
    )


def calculate_jiwer_counts(reference: str, predicted: str) -> tuple[int, int, int, int]:
    """Calculate normalized edit counts and reference lengths for jiwer metrics.

    This returns the counts needed to aggregate per-page jiwer results by
    summing edit distances and normalized reference lengths rather than
    concatenating pages into one document.

    Applies the same custom transforms as calculate_jiwer_metrics():
    - ToLowerCase()
    - RemovePunctuation()
    - Strip()
    - RemoveMultipleSpaces() (for WER only)

    The aggregation strategy is:
    1) normalize each page independently, then
    2) sum normalized edit distances and reference lengths across pages,
    3) compute the final rate from the totals.
    """
    character_output = process_characters(
        reference,
        predicted,
        reference_transform=cer_custom_transform,
        hypothesis_transform=cer_custom_transform,
    )
    word_output = process_words(
        reference,
        predicted,
        reference_transform=wer_custom_transform,
        hypothesis_transform=wer_custom_transform,
    )

    char_edits = (
        character_output.substitutions
        + character_output.deletions
        + character_output.insertions
    )
    ref_chars = sum(len(sentence) for sentence in character_output.references)

    word_edits = (
        word_output.substitutions + word_output.deletions + word_output.insertions
    )
    ref_words = sum(len(sentence) for sentence in word_output.references)

    return char_edits, ref_chars, word_edits, ref_words


def _count_char_errors_from_editops(ref: str, hyp: str) -> Counter:
    """Count errors attributed to reference characters using Levenshtein editops.

    Substitutions and deletions are attributed to the reference character that
    was not correctly recognized. Insertions are not attributed to any
    reference character (they are extra predicted characters).
    Returns a Counter mapping reference characters -> error counts.
    """
    counts: Counter = Counter()
    ops = Levenshtein.editops(ref, hyp)
    for op, i, _ in ops:
        if op == "replace":
            # substitution: ref char at i replaced by hyp char at j
            if i < len(ref):
                counts[ref[i]] += 1
        elif op == "delete":
            # deletion: ref char at i was deleted in hypothesis
            if i < len(ref):
                counts[ref[i]] += 1
        # insertions are not attributed to a reference char
    return counts


def top_error_chars_for_pair(
    reference: str, predicted: str, normalize: bool = False, top_n: int = 10
) -> List[Tuple[str, int]]:
    """Return top N reference characters that caused errors for a single pair.

    If `normalize` is True, the function applies the same jiwer character
    normalization pipeline used for CER (`cer_custom_transform`) and
    aggregates counts per normalized sentence before alignment. This makes
    the result comparable to the normalized CER regime.
    """
    if not normalize:
        counts = _count_char_errors_from_editops(reference, predicted)
    else:
        # use jiwer to get normalized character-level sentences and align each
        # normalized sentence pair separately to avoid cross-sentence joins
        char_output = process_characters(
            reference,
            predicted,
            reference_transform=cer_custom_transform,
            hypothesis_transform=cer_custom_transform,
        )
        counts = Counter()
        # zip through sentences; jiwer guarantees parallel lists
        for ref_sent, hyp_sent in zip(char_output.references, char_output.hypotheses):
            ref_str = "".join(ref_sent)
            hyp_str = "".join(hyp_sent)
            counts.update(_count_char_errors_from_editops(ref_str, hyp_str))

    most_common = counts.most_common(top_n)
    return most_common


def aggregate_top_error_chars(
    references: Iterable[str],
    predictions: Iterable[str],
    normalize: bool = False,
    top_n: int = 10,
) -> List[Tuple[str, int, float]]:
    """Aggregate character error counts across multiple reference/prediction pairs.

    Returns a list of tuples: (character, count, percent_of_all_ref-errors).
    Only substitutions and deletions (i.e., errors that map to a reference
    character) are considered. `percent_of_all_ref-errors` is relative to the
    total number of reference-attributed errors.
    """
    total_counts: Counter = Counter()
    for ref, hyp in zip(references, predictions):
        if not normalize:
            total_counts.update(_count_char_errors_from_editops(ref, hyp))
        else:
            char_output = process_characters(
                ref,
                hyp,
                reference_transform=cer_custom_transform,
                hypothesis_transform=cer_custom_transform,
            )
            for ref_sent, hyp_sent in zip(
                char_output.references, char_output.hypotheses
            ):
                ref_str = "".join(ref_sent)
                hyp_str = "".join(hyp_sent)
                total_counts.update(_count_char_errors_from_editops(ref_str, hyp_str))

    total_errors = sum(total_counts.values())
    results: List[Tuple[str, int, float]] = []
    for char, cnt in total_counts.most_common(top_n):
        pct = (cnt / total_errors * 100) if total_errors > 0 else 0.0
        results.append((char, cnt, pct))
    return results
