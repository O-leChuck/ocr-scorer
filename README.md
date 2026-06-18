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
   - Uses `jiwer` [default transforms](https://jitsi.github.io/jiwer/reference/transformations/)
   - Applies normalization before error calculation
   - This includes lowercasing, punctuation removal, and whitespace normalization
   - Calculates:
     - CER normalized
     - WER normalized

## Files

- `main.py` - main scoring script
- `utils.py` - folder selection helper functions
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

### jiwer normalized evaluation

- Uses `jiwer` default normalization pipeline.
- Use this regime when you want a more forgiving and standardized OCR evaluation.

## Example

Raw whitespace evaluation counts token-level differences directly, while normalized evaluation first applies text transforms to reduce tokenization artifacts.

| Case | Reference | Prediction | Raw white-space WER behavior | Normalized jiwer behavior |
|---|---|---|---|---|
| Word split | `I found a fountain.` | `I found a foun tain.` | May count as 2 word errors because prediction becomes `foun` + `tain.` | Counts the underlying split as a single transcript error after normalization |
| Capitalization | `Hello World.` | `hello world.` | Counts as 2 word substitutions if exact-match is required | Normalizes case, so the words match |
| Punctuation | `Hello, world!` | `Hello world` | May count as 2 word differences because punctuation is part of tokens | Removes punctuation before matching |
| Hyphenation | `high-quality results` | `high quality results` | May count as one deletion and one insertion or substitution | Normalizes hyphenation and is less likely to penalize it as two errors |
| Expected metric effect | - | - | Sensitive to formatting/tokenization; may overcount errors | Focuses on actual transcription differences; less sensitive to formatting |

### Expanded normalization example

Reference:

```text
This is a high-quality result.
It contains punctuation, capitalization, and line breaks.
```

Prediction:

```text
this is a high quality result
it contains punctuation capitalization and line breaks
```

- Raw whitespace evaluation treats punctuation and casing as token-level mismatches.
- The normalized `jiwer` regime lowercases text, removes punctuation, and normalizes whitespace.
- That means the normalized regime measures the real transcription differences, not formatting differences.

## Notes

- The raw regime is useful for exact whitespace/tokenization-sensitive comparisons.
- The normalized regime is useful for standard OCR/speech recognition comparisons where punctuation and casing should not inflate the error rate.
