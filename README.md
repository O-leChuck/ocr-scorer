# OCR Scorer

This script computes OCR evaluation metrics between ground truth text files and predicted OCR text files.

## What it calculates

The script supports two evaluation regimes:

1. **Raw whitespace evaluation**
   - Uses Python `split()` on raw text
   - No normalization, no lowercasing, no punctuation removal
   - Calculates:
     - CER raw
     - WER raw

2. **Normalized jiwer evaluation**
   - Uses a custom `jiwer` transform pipeline with lowercasing and punctuation handling
   - Punctuation is converted to spaces for WER so word boundaries are preserved
   - Applies aggressive normalization before error calculation
   - See [Normalization Details](#normalization-details) below for exact transforms
   - Calculates:
     - CER normalized
     - WER normalized

## Files

- `main.py` - main scoring script
- `io_helpers.py` - folder selection and metric export helper functions
- `requirements.txt` - Python package requirements
- `README.md` - usage and documentation

## Usage

1. Create and activate a virtual environment

2. Install dependencies if needed:

```bash
pip install -r requirements.txt
```

3. Run the script:

```bash
python main.py
```

4. Select the GT folder and the prediction folder when prompted.

## Output

The script saves:

- `metrics_pagewise.csv`
- `metrics_pagewise.json`
- `metrics_visualization.png`

These files are written to the parent directory of the selected prediction folder.

## Metric details

### Raw whitespace evaluation

- Uses direct whitespace splitting of both reference and predicted text.
- This is a strict raw WER implementation.
- It is sensitive to punctuation, capitalization, and tokenization differences.
- **CER Raw:** Levenshtein distance on raw characters / total raw reference characters
- **WER Raw:** Levenshtein distance on whitespace-split words / total raw reference words

### Normalized jiwer evaluation

- Uses a custom `jiwer` transform pipeline with lowercasing and punctuation removal.
- **Aggregation method:** Sums character/word edit counts across all pages, then divides by total normalized reference characters/words. This approach is suitable for randomly-sampled page-level evaluation.
- **CER Normalized:** Sum of normalized character edits / sum of normalized reference characters
- **WER Normalized:** Sum of normalized word edits / sum of normalized reference words

### Normalization Details

The script currently applies a custom jiwer transform pipeline for both CER and WER. The following transforms are used:

1. **ToLowerCase()** - Converts all text to lowercase
2. **RemovePunctuation()** - Removes all Unicode punctuation (categories Po, Pd, Ps, Pe, Pi, Pf, Pc)
3. **Strip()** - Removes leading and trailing whitespace

#### Character Error Rate (CER) Normalization

After the common transforms, the script applies:

1. **ReduceToListOfListOfChars()** - Converts the normalized text to a list of characters for CER calculation

#### Word Error Rate (WER) Normalization

After the common transforms, the script applies:

1. **RemoveMultipleSpaces()** - Collapses multiple consecutive spaces into a single space
2. **ReduceToListOfListOfWords()** - Splits the normalized text into words using whitespace tokenization

#### What punctuation is removed?

`RemovePunctuation()` strips characters whose Unicode category begins with `P`, including:
- ASCII punctuation: `! " # % & ' ( ) * + , - . / : ; < = > ? @ [ \ ] ^ _ ` { | } ~`
- Dash punctuation: `-`, `–`, `—`, `‑`, etc.
- Quotes and brackets: `"`, `''`, `«`, `»`, `‹`, `›`, `(`, `)`, `[`, `]`, `{`, `}`
- Many historic punctuation marks and typographic symbols that are classified in Unicode as punctuation

It does not remove symbols that are not classified as punctuation by Unicode, such as currency signs, math operators, or letter-like marks.

This means the current normalized regime is more aggressive than plain whitespace normalization: it lowercases text and removes punctuation while still counting real additions, deletions, and substitutions.

If the exact shape of historic punctuation is important, preserve it by removing `RemovePunctuation()` from the custom transform pipeline in `metrics.py`.

#### Why normalized may still differ from raw only a little

If your raw and normalized results are similar, it likely means:
- Your OCR output already has consistent formatting and spacing
- The remaining errors are actual transcription mistakes rather than punctuation/case differences
- The pipeline removes punctuation and lowercases text, but still counts real additions, deletions, and substitutions

If you want a different normalization behavior, you can modify `metrics.py` to adjust the jiwer transform pipeline.

## Example: Raw vs. Normalized

| Case | Reference | Prediction | Raw WER | Normalized WER | Why? |
|---|---|---|---|---|---|
| Multiple spaces | `word1  word2` | `word1 word2` | 0 words match exactly | 0 words match exactly | RemoveMultipleSpaces normalizes both → same word list |
| Leading/trailing spaces | ` word ` | `word` | Counts as different if exact string match used | Counts as match | Strip() removes spaces before comparison |
| Capitalization | `Hello` | `hello` | Word mismatch | Match | Lowercasing makes the words equal |
| Punctuation | `Hello.` | `Hello` | Word mismatch | Match | Punctuation removal makes the tokens equal |
| Hyphenation | `high-quality` | `high quality` | 2 different words | 2 different words | Hyphenation is not normalized; the words remain different |

### When normalized and raw differ

The normalized metrics primarily help with:
- **Multiple/irregular spacing** - RemoveMultipleSpaces collapses consecutive spaces
- **Whitespace trimming** - Strip() removes leading/trailing spaces on each page
- **Consistent tokenization** - Ensures word boundaries are based on single spaces

The normalized metrics will be similar to raw if:
- OCR output already has consistent, single-space word separation
- Most errors are genuine character/word-level mistakes, not formatting issues
- Text does not have irregular leading/trailing spaces

### Customizing normalization

The current script already applies a custom jiwer pipeline with lowercasing, punctuation removal, whitespace normalization, and page-level aggregation.

To change this behavior, edit [metrics.py](metrics.py) and adjust the `cer_custom_transform` and `wer_custom_transform` pipelines.

For example, to preserve punctuation for historic documents, remove `RemovePunctuation()` from the pipeline:

```python
from jiwer import Compose, ToLowerCase, Strip, RemoveMultipleSpaces, ReduceToListOfListOfWords

wer_custom_transform = Compose([
    ToLowerCase(),
    RemoveMultipleSpaces(),
    Strip(),
    ReduceToListOfListOfWords(),
])
```

Then pass the custom transforms to `jiwer.wer()` and `jiwer.cer()` via `reference_transform` and `hypothesis_transform`.

## Notes

- The raw regime is useful for exact whitespace/tokenization-sensitive comparisons.
- The normalized regime now applies lowercasing and punctuation removal in addition to whitespace normalization.
- If normalized and raw results are very similar, your OCR output likely has consistent formatting and the remaining errors are actual transcription differences.
- Modify `metrics.py` if you want a different normalization pipeline.
