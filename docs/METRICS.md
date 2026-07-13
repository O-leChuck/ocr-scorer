# Metrics reference

This document is the detailed reference for exactly how OCR Scorer computes
CER and WER, and exactly how edge cases (like an empty ground-truth page)
are represented in each output file. For "how do I run the tool", see the
[main README](../README.md) instead - this page assumes you already have
results and want to understand what they mean.

## Contents

- [Raw whitespace evaluation](#raw-whitespace-evaluation)
- [Normalized jiwer evaluation](#normalized-jiwer-evaluation)
- [Normalization details](#normalization-details)
- [Example: raw vs. normalized](#example-raw-vs-normalized)
- [When normalized and raw differ](#when-normalized-and-raw-differ)
- [Customizing normalization](#customizing-normalization)
- [Empty-reference pages: exact representation](#empty-reference-pages-exact-representation)

## Raw whitespace evaluation

- Uses direct whitespace splitting of both reference and predicted text.
- This is a strict raw WER implementation.
- It is sensitive to punctuation, capitalization, and tokenization differences.
- **CER Raw:** Levenshtein distance on raw characters / total raw reference characters
- **WER Raw:** Levenshtein distance on whitespace-split words / total raw reference words

## Normalized jiwer evaluation

- Uses a custom `jiwer` transform pipeline with lowercasing and punctuation handling (removed for CER, replaced with a space for WER - see [Normalization details](#normalization-details)).
- **Aggregation method:** Sums character/word edit counts across all pages, then divides by total normalized reference characters/words. This approach is suitable for randomly-sampled page-level evaluation.
- **CER Normalized:** Sum of normalized character edits / sum of normalized reference characters
- **WER Normalized:** Sum of normalized word edits / sum of normalized reference words

## Normalization details

CER and WER use similar but not identical transform pipelines - the key
difference is how each one handles punctuation.

### Character Error Rate (CER) normalization

1. **ToLowerCase()** - Converts all text to lowercase
2. **RemovePunctuation()** - Removes all Unicode punctuation entirely (categories Po, Pd, Ps, Pe, Pi, Pf, Pc)
3. **Strip()** - Removes leading and trailing whitespace
4. **ReduceToListOfListOfChars()** - Converts the normalized text to a list of characters for CER calculation

### Word Error Rate (WER) normalization

1. **ToLowerCase()** - Converts all text to lowercase
2. **ReplacePunctuationWithSpace()** - Replaces Unicode punctuation with a space, instead of removing it, so that punctuation-separated words don't get merged together (`"high-quality"` → `"high quality"`, not `"highquality"`)
3. **RemoveMultipleSpaces()** - Collapses multiple consecutive spaces into a single space (this also cleans up any doubled-up spaces the previous step may have introduced)
4. **Strip()** - Removes leading and trailing whitespace
5. **ReduceToListOfListOfWords()** - Splits the normalized text into words using whitespace tokenization

This means punctuation that sits *between* words (like a hyphen) affects CER and WER differently: WER treats the two sides as separate words that can still match, while CER only sees the punctuation being deleted, which is now a genuine character-level difference.

### What punctuation is affected?

Both transforms act on characters whose Unicode category begins with `P`, including:
- ASCII punctuation: `! " # % & ' ( ) * + , - . / : ; < = > ? @ [ \ ] ^ _ ` { | } ~`
- Dash punctuation: `-`, `–`, `—`, `‑`, etc.
- Quotes and brackets: `"`, `''`, `«`, `»`, `‹`, `›`, `(`, `)`, `[`, `]`, `{`, `}`
- Many historic punctuation marks and typographic symbols that are classified in Unicode as punctuation

Neither transform touches symbols that are not classified as punctuation by Unicode, such as currency signs, math operators, or letter-like marks.

This means the current normalized regime is more aggressive than plain whitespace normalization: it lowercases text and strips/converts punctuation while still counting real additions, deletions, and substitutions.

If the exact shape of historic punctuation is important to your evaluation, preserve it by removing `RemovePunctuation()` (for CER) and/or `ReplacePunctuationWithSpace()` (for WER) from the custom transform pipelines in `ocr_scorer/metrics.py`.

### Why normalized may still differ from raw only a little

If your raw and normalized results are similar, it likely means:
- Your OCR output already has consistent formatting and spacing
- The remaining errors are actual transcription mistakes rather than punctuation/case differences
- The pipeline removes/converts punctuation and lowercases text, but still counts real additions, deletions, and substitutions

## Example: raw vs. normalized

| Case | Reference | Prediction | Raw WER | Normalized WER | Why? |
|---|---|---|---|---|---|
| Multiple spaces | `word1  word2` | `word1 word2` | 0 words match exactly | 0 words match exactly | RemoveMultipleSpaces normalizes both → same word list |
| Leading/trailing spaces | ` word ` | `word` | Counts as different if exact string match used | Counts as match | Strip() removes spaces before comparison |
| Capitalization | `Hello` | `hello` | Word mismatch | Match | Lowercasing makes the words equal |
| Punctuation | `Hello.` | `Hello` | Word mismatch | Match | Punctuation removal makes the tokens equal |
| Hyphenation | `high-quality` | `high quality` | 2 different words | Match (0% WER) | `ReplacePunctuationWithSpace()` turns the hyphen into a space, so both sides tokenize to `["high", "quality"]`. Note CER is *not* fully normalized here (~9% CER remains), since CER's `RemovePunctuation()` deletes the hyphen instead of replacing it with a space, leaving `"highquality"` vs `"high quality"`. |

## When normalized and raw differ

The normalized metrics primarily help with:
- **Multiple/irregular spacing** - RemoveMultipleSpaces collapses consecutive spaces
- **Whitespace trimming** - Strip() removes leading/trailing spaces on each page
- **Consistent tokenization** - Ensures word boundaries are based on single spaces

The normalized metrics will be similar to raw if:
- OCR output already has consistent, single-space word separation
- Most errors are genuine character/word-level mistakes, not formatting issues
- Text does not have irregular leading/trailing spaces

## Customizing normalization

The current script already applies a custom jiwer pipeline with lowercasing, punctuation handling, whitespace normalization, and page-level aggregation.

To change this behavior, edit [ocr_scorer/metrics.py](../ocr_scorer/metrics.py) and adjust the `cer_custom_transform` and `wer_custom_transform` pipelines.

For example, to preserve punctuation for historic documents, drop the punctuation step from each pipeline:

```python
from jiwer import Compose, ToLowerCase, Strip, RemoveMultipleSpaces, ReduceToListOfListOfWords, ReduceToListOfListOfChars

cer_custom_transform = Compose([
    ToLowerCase(),
    Strip(),
    ReduceToListOfListOfChars(),
])

wer_custom_transform = Compose([
    ToLowerCase(),
    RemoveMultipleSpaces(),
    Strip(),
    ReduceToListOfListOfWords(),
])
```

These are the same `cer_custom_transform`/`wer_custom_transform` objects already used by `calculate_jiwer_metrics()` and `calculate_jiwer_counts()` in `ocr_scorer/metrics.py`, so editing them there is enough - no other code needs to change.

## Empty-reference pages: exact representation

If a ground-truth page is empty but the OCR still produced text (a
hallucination), that page's CER/WER can't be expressed as a normal
percentage - the calculation is a genuine division by zero. Rather than
disguise this with a made-up number, the tool reports it explicitly as
undefined/infinite rather than a fake percentage or a silent gap. This is
deliberate: capping or hiding it would hide exactly the failure mode
(hallucinating readable text onto blank pages) that CER/WER is meant to
catch.

- **Reference empty, prediction non-empty** (hallucination): a genuine
  divide-by-zero. Mathematically, this is the limit of a positive number
  divided by an ever-shrinking denominator, i.e. infinite - not "undefined"
  in the sense of being arbitrary, just unbounded. It is deliberately not
  capped or excluded.
- **Reference and prediction both empty**: a true 0/0 - there is nothing
  to measure on that page (no error, since nothing was predicted for
  nothing expected). Reported as not-a-number/not-applicable rather than a
  fabricated 0% or 100%.
- **CSV** (`metrics_pagewise.csv`, `metrics_document_summary.csv`):
  infinite values are written as the literal text `inf`; not-applicable
  values are left blank. Both round-trip correctly as real numbers in
  pandas/numpy/R. Excel does not recognize the text `inf` as numeric, so a
  plain `=AVERAGE()` over such a column will silently skip that cell in
  Excel specifically - though the word `inf` sitting in a numeric column
  is still a strong visual flag for a human reader, even where Excel's own
  formulas don't propagate it the way Python/R do.
- **JSON** (`metrics_pagewise.json`, `metrics_document.json`): infinite
  values are written as the string `"Infinity"`/`"-Infinity"` (JSON has no
  native infinity token) rather than `null`, specifically so they can't be
  mistaken for ordinary missing data - this does mean that field is no
  longer purely numeric for that one page. Not-applicable (0/0) values are
  written as `null`.
- **Chart/PDF** (`metrics_visualization.png`, `evaluation_report.pdf`): an
  infinite page shows as a break in the line, flagged with a red triangle
  marker at the top of the chart, rather than distorting the whole
  chart's scale or silently plotting off the visible area.
- This all applies to the **raw** (whitespace) CER/WER, which is entirely
  our own calculation. The **jiwer-normalized** CER/WER for an individual
  page still goes through the third-party `jiwer` library's own internal
  handling of empty-after-normalization text, which currently returns a
  specific (if unintuitive) finite value rather than infinity - see
  [Normalization details](#normalization-details). The jiwer-normalized
  *document-wide* aggregate, however, is our own summation and follows the
  same infinity/not-applicable convention as the raw metrics.
