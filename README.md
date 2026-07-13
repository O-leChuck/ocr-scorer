# OCR Scorer

This script computes OCR evaluation metrics (Character Error Rate and Word Error Rate) between ground truth text files and prediction text files from OCR systems.

A limited browser-based version of the OCR Scorer is also available in the `webapp` folder. It calculates only the naive case-sensitive CER and WER values, but it can be opened directly in any modern browser and used without installing Python or writing code.

## What it calculates

The script supports two evaluation regimes:

1. **Raw whitespace evaluation**
   - No normalization, no lowercasing, no punctuation removal
   - Uses Python `split()` on raw text to separate words, de facto removing whitespace of any kind, leaving it out of the WER evaluation
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

- `main.py` - thin compatibility entry point (`python main.py`); the
  actual implementation lives in `ocr_scorer/`
- `ocr_scorer/evaluate.py` - the core evaluation logic (`run_evaluation()`),
  the intended import point for using this as a library/pipeline step
- `ocr_scorer/cli.py` - interactive/command-line entry point
- `ocr_scorer/dialogs.py` - the interactive folder-selection dialogs (the
  only module that touches tkinter)
- `ocr_scorer/io_helpers.py` - metric/log export and page-pairing helpers
- `ocr_scorer/metrics.py` - CER/WER calculation and character-error analysis
- `ocr_scorer/plotting.py` - chart and PDF report generation
- `ocr_scorer/config.py` - loads optional default folder paths from `config.ini`
- `config.template.ini` - template for `config.ini` (see [Configuring default folders](#configuring-default-folders))
- `requirements.txt` - Python package requirements
- `README.md` - usage and documentation

## Requirements

- Python 3.10 or newer

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

Alternatively, you can install the tool itself (`pip install -e .`), which
also gives you an `ocr-scorer` command you can run from anywhere.

4. When prompted, select **any one `.txt` file inside** your Goldstandard
   (ground truth) folder, then do the same for your predictions folder -
   the tool uses that file's parent folder. This is deliberate: the dialog
   is a regular file picker (not a plain folder picker), so you can see the
   folder's contents while browsing and immediately notice if you've
   navigated into the wrong or an empty folder.

   Both folders must contain the same number of `.txt` files - if the
   counts don't match, you'll get an error dialog and a chance to pick
   again. Both selected paths are also printed to the terminal as soon as
   you pick them, so you can double-check them right away.

### Configuring default folders

The folder picker needs a starting directory. By default it tries, in
order:

1. A path you configure yourself in `config.ini` (see below) - the
   recommended option once you have a regular working folder.
2. Your home directory, as a machine-agnostic, always-exists fallback
   that works the same way on Windows/macOS/Linux and makes no
   assumption about your folder structure.

Whichever one is used, it's printed to the terminal so it's never a silent
guess. To set your own default: copy `config.template.ini` to `config.ini`
(same folder) and fill in `goldstandard_folder`/`prediction_folder` under
`[paths]`. `config.ini` is in `.gitignore`, so your local paths are never
committed. Either entry can be left blank; an invalid or nonexistent path
is ignored (with a warning) rather than breaking the tool, falling through
to the next option in the list above.

An earlier version of this tool also tried to auto-detect these folders by
searching your entire home directory for folders named
`Goldstandard`/`Lumen-Lucernae`. That was removed: it silently picked
whichever matching folder it found first, which could be the wrong one if
more than one folder shared that name anywhere under your home directory
(for example, a stray copy nested inside an old prediction run's output
folder) - with no way to guarantee that couldn't happen. `config.ini`
avoids this ambiguity entirely, since it points at an exact folder rather
than searching for a name.

### Important: how pages are matched

The script pairs up ground-truth and prediction files by **sorted position**,
not by filename - the first (alphabetically sorted) `.txt` file in the GT
folder is compared against the first sorted `.txt` file in the prediction
folder, and so on. Filenames themselves can differ between the two folders.

This means you must make sure that:

- Both folders contain exactly one file per page, with no pages missing or
  duplicated.
- Files sort into the same page order in both folders. Zero-padded page
  numbers (`page_001.txt`, `page_002.txt`, ...) are a safe way to ensure
  this; `page_2.txt` sorting before `page_10.txt` is a common source of
  silent misalignment.

If a page is missing from one folder while an unrelated extra file exists in
the other, the file *counts* can still match while the actual page pairing
is wrong - the tool has no way to detect this, so double-check your input
folders before relying on the results.

### Non-interactive / scripted use

For running this as a step in another pipeline (rather than by hand), there
are two options that skip the folder-picker dialogs entirely:

**From the command line**, pass both folders as flags:

```bash
python main.py --gt /path/to/goldstandard --pred /path/to/predictions
```

`--gt` and `--pred` must be given together; providing only one is treated
as a usage error rather than falling back to the interactive dialog for
the other.

**From another Python script**, import `run_evaluation()` from the
`ocr_scorer` package directly, instead of going through the CLI:

```python
from ocr_scorer import run_evaluation

output_dir, document_metrics = run_evaluation(gt_path, pred_path)
```

`import ocr_scorer` never touches tkinter, so this works in headless
environments (servers, CI, minimal Docker images) where a display - or
even tkinter itself - isn't available; only the interactive CLI
(`ocr_scorer.cli`/`ocr_scorer.dialogs`) needs it.

`run_evaluation()` runs the same evaluation as the interactive tool -
writing the same files to a dated `evaluation_YYYY-MM-DD` folder next to
the prediction folder (see [Output](#output)) - and additionally returns
the path to that folder plus the document-wide summary dict, so a calling
pipeline can act on the results without re-reading files from disk.

Unlike the interactive flow, there's no dialog to retry folder selection
if something's wrong, so `run_evaluation()` raises `ValueError` instead
(rather than printing a message and silently doing nothing) if either
folder doesn't exist, contains no `.txt` files, or the two folders contain
different numbers of files. The CLI catches this and prints a plain error
message when run from the command line.

Anything noteworthy that happens during a run but doesn't stop it (a page
that failed to read, a page whose CER/WER calculation itself failed and
was skipped so the rest of the batch still completes, a page-number
mismatch, a page whose CER/WER came out undefined - see
[Empty-reference pages](#empty-reference-pages)) is by default also
printed to the terminal, which a pipeline could easily miss. Every such
message is therefore also collected into
`document_metrics["warnings"]`, so a caller can check for problems without
watching stdout. Pass `verbose=False` to suppress the printing entirely
(e.g. if your pipeline has its own logging) without losing anything - the
same information is still in `document_metrics["warnings"]`.

## Output

Each run creates a new, dated subfolder next to the selected prediction
folder, named `evaluation_YYYY-MM-DD` (using the current date). Results are
never overwritten in place, so you can re-run the tool multiple times per
day/dataset without losing earlier results (running it twice on the same day
does overwrite that day's folder, though).

That folder contains:

- `evaluation_log.txt` - which GT/prediction folders were used, the
  evaluation date, and the list of files that were scored
- `metrics_pagewise.csv` / `metrics_pagewise.json` - CER/WER per page, both
  raw and jiwer-normalized
- `metrics_document.json` - a single nested summary: overall CER/WER plus
  the top recurring error characters (see below)
- `metrics_document_summary.csv` - the document-wide CER/WER summary as a
  flat table
- `metrics_document_top_raw.csv` / `metrics_document_top_normalized.csv` -
  the characters most often misrecognized, raw and normalized (only written
  if there are any attributable errors)
- `metrics_visualization.png` - a chart of CER/WER per page
- `evaluation_report.pdf` - a one-file report combining the summary, the
  top-error tables, and the chart, suitable for sharing or archiving

### About the "top error characters" report

This report counts, per reference character, how often it was substituted
for the wrong character or dropped entirely (deletions), then ranks the 10
most frequent offenders. Two caveats:

- Only alphabetic reference characters are counted - digits, punctuation,
  and other symbols are excluded, even if the OCR got them wrong. If you're
  evaluating heavily numeric or symbol-heavy material, this report won't
  reflect those errors.
- Extra characters the OCR *inserted* that don't correspond to any ground
  truth character aren't attributed to anything, since there's no
  reference character to blame them on.

### Empty-reference pages

If a ground-truth page is empty but the OCR still produced text (a
hallucination), that page's CER/WER can't be expressed as a normal
percentage - the calculation is a genuine division by zero. Rather than
disguise this with a made-up number, the script reports it explicitly as
**undefined/infinite**, and shows it in the chart/PDF as a gap in the line
with a red "undefined" marker at the top, instead of a fake percentage or
a silent gap. This is deliberate: capping or hiding it would hide exactly
the failure mode (hallucinating readable text onto blank pages) that
CER/WER is meant to catch. See the footnote at the bottom of this
document for exactly how this value is represented in each output file.

## Metric details

### Raw whitespace evaluation

- Uses direct whitespace splitting of both reference and predicted text.
- This is a strict raw WER implementation.
- It is sensitive to punctuation, capitalization, and tokenization differences.
- **CER Raw:** Levenshtein distance on raw characters / total raw reference characters
- **WER Raw:** Levenshtein distance on whitespace-split words / total raw reference words

### Normalized jiwer evaluation

- Uses a custom `jiwer` transform pipeline with lowercasing and punctuation handling (removed for CER, replaced with a space for WER - see [Normalization Details](#normalization-details)).
- **Aggregation method:** Sums character/word edit counts across all pages, then divides by total normalized reference characters/words. This approach is suitable for randomly-sampled page-level evaluation.
- **CER Normalized:** Sum of normalized character edits / sum of normalized reference characters
- **WER Normalized:** Sum of normalized word edits / sum of normalized reference words

### Normalization Details

CER and WER use similar but not identical transform pipelines - the key
difference is how each one handles punctuation.

#### Character Error Rate (CER) Normalization

1. **ToLowerCase()** - Converts all text to lowercase
2. **RemovePunctuation()** - Removes all Unicode punctuation entirely (categories Po, Pd, Ps, Pe, Pi, Pf, Pc)
3. **Strip()** - Removes leading and trailing whitespace
4. **ReduceToListOfListOfChars()** - Converts the normalized text to a list of characters for CER calculation

#### Word Error Rate (WER) Normalization

1. **ToLowerCase()** - Converts all text to lowercase
2. **ReplacePunctuationWithSpace()** - Replaces Unicode punctuation with a space, instead of removing it, so that punctuation-separated words don't get merged together (`"high-quality"` → `"high quality"`, not `"highquality"`)
3. **RemoveMultipleSpaces()** - Collapses multiple consecutive spaces into a single space (this also cleans up any doubled-up spaces the previous step may have introduced)
4. **Strip()** - Removes leading and trailing whitespace
5. **ReduceToListOfListOfWords()** - Splits the normalized text into words using whitespace tokenization

This means punctuation that sits *between* words (like a hyphen) affects CER and WER differently: WER treats the two sides as separate words that can still match, while CER only sees the punctuation being deleted, which is now a genuine character-level difference.

#### What punctuation is affected?

Both transforms act on characters whose Unicode category begins with `P`, including:
- ASCII punctuation: `! " # % & ' ( ) * + , - . / : ; < = > ? @ [ \ ] ^ _ ` { | } ~`
- Dash punctuation: `-`, `–`, `—`, `‑`, etc.
- Quotes and brackets: `"`, `''`, `«`, `»`, `‹`, `›`, `(`, `)`, `[`, `]`, `{`, `}`
- Many historic punctuation marks and typographic symbols that are classified in Unicode as punctuation

Neither transform touches symbols that are not classified as punctuation by Unicode, such as currency signs, math operators, or letter-like marks.

This means the current normalized regime is more aggressive than plain whitespace normalization: it lowercases text and strips/converts punctuation while still counting real additions, deletions, and substitutions.

If the exact shape of historic punctuation is important to your evaluation, preserve it by removing `RemovePunctuation()` (for CER) and/or `ReplacePunctuationWithSpace()` (for WER) from the custom transform pipelines in `ocr_scorer/metrics.py`.

#### Why normalized may still differ from raw only a little

If your raw and normalized results are similar, it likely means:
- Your OCR output already has consistent formatting and spacing
- The remaining errors are actual transcription mistakes rather than punctuation/case differences
- The pipeline removes/converts punctuation and lowercases text, but still counts real additions, deletions, and substitutions

If you want a different normalization behavior, you can modify `ocr_scorer/metrics.py` to adjust the jiwer transform pipelines.

## Example: Raw vs. Normalized

| Case | Reference | Prediction | Raw WER | Normalized WER | Why? |
|---|---|---|---|---|---|
| Multiple spaces | `word1  word2` | `word1 word2` | 0 words match exactly | 0 words match exactly | RemoveMultipleSpaces normalizes both → same word list |
| Leading/trailing spaces | ` word ` | `word` | Counts as different if exact string match used | Counts as match | Strip() removes spaces before comparison |
| Capitalization | `Hello` | `hello` | Word mismatch | Match | Lowercasing makes the words equal |
| Punctuation | `Hello.` | `Hello` | Word mismatch | Match | Punctuation removal makes the tokens equal |
| Hyphenation | `high-quality` | `high quality` | 2 different words | Match (0% WER) | `ReplacePunctuationWithSpace()` turns the hyphen into a space, so both sides tokenize to `["high", "quality"]`. Note CER is *not* fully normalized here (~9% CER remains), since CER's `RemovePunctuation()` deletes the hyphen instead of replacing it with a space, leaving `"highquality"` vs `"high quality"`. |

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

The current script already applies a custom jiwer pipeline with lowercasing, punctuation handling, whitespace normalization, and page-level aggregation.

To change this behavior, edit [ocr_scorer/metrics.py](ocr_scorer/metrics.py) and adjust the `cer_custom_transform` and `wer_custom_transform` pipelines.

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

## Notes

- The raw regime is useful for exact whitespace/tokenization-sensitive comparisons.
- The normalized regime applies lowercasing and punctuation handling (removed for CER, replaced with a space for WER) in addition to whitespace normalization.
- If normalized and raw results are very similar, your OCR output likely has consistent formatting and the remaining errors are actual transcription differences.
- Modify `ocr_scorer/metrics.py` if you want a different normalization pipeline.

### Footnote: exactly how empty-reference values are represented

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
  [Normalization Details](#normalization-details). The jiwer-normalized
  *document-wide* aggregate, however, is our own summation and follows the
  same infinity/not-applicable convention as the raw metrics.

## License

MIT - see [LICENSE](LICENSE).
