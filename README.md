# OCR Scorer

A tool for computing OCR evaluation metrics - Character Error Rate (CER)
and Word Error Rate (WER) - between ground-truth text files and OCR
prediction text files. Aimed at digital humanities researchers evaluating
OCR on historical documents.

A limited browser-based version is also available in the `webapp` folder.
It calculates only the naive, case-sensitive CER and WER, but runs
directly in any modern browser - no Python or command line needed.

## Contents

- [Quick start](#quick-start)
- [What it calculates](#what-it-calculates)
- [Usage](#usage)
  - [Interactive use](#interactive-use)
  - [Configuring default folders](#configuring-default-folders)
  - [Important: how pages are matched](#important-how-pages-are-matched)
  - [Non-interactive / scripted use](#non-interactive--scripted-use)
- [Output](#output)
- [Project layout](#project-layout)
- [Requirements](#requirements)
- [License](#license)

## Quick start

```bash
# 1. Create and activate a virtual environment, then install dependencies
pip install -r requirements.txt

# 2. Run it
python main.py
```

You'll be prompted to pick your ground truth folder, then
your predictions folder - see [Interactive use](#interactive-use) for what
exactly to click. Results are written next to your predictions folder; see
[Output](#output).

Alternatively, install the tool itself (`pip install -e .`), which also
gives you an `ocr-scorer` command you can run from any directory:

```bash
pip install -e .
ocr-scorer
```

## What it calculates

Every run produces two independent sets of CER/WER numbers:

1. **Raw** - no normalization at all: exact character comparison, and
   words split on whitespace only. Sensitive to case, punctuation, and
   spacing differences.
2. **Normalized** - lowercased, with punctuation removed (CER) or turned
   into a space (WER), via a `jiwer` transform pipeline.

For the exact transform pipelines, worked examples of raw vs. normalized
results, and how to customize the normalization, see
[docs/METRICS.md](docs/METRICS.md).

## Usage

### Interactive use

Run `python main.py` (or `ocr-scorer` if installed). When prompted, select
**any one `.txt` file inside** your ground truth folder, then do the same
for your predictions folder - the tool uses that file's parent folder.
This is deliberate: the dialog is a regular file picker (not a plain
folder picker), so you can see the folder's contents while browsing and
immediately notice if you've navigated into the wrong or an empty folder.

Both folders must contain the same number of `.txt` files - if the counts
don't match, you'll get an error dialog and a chance to pick again. Both
selected paths are also printed to the terminal as soon as you pick them,
so you can double-check them right away.

### Configuring default folders

The folder picker needs a starting directory. By default it tries, in
order:

1. A path you configure yourself in `config.ini` (see below) - the
   recommended option once you have a regular working folder.
2. Your home directory, as a machine-agnostic, always-exists fallback
   that works the same way on Windows/macOS/Linux and makes no assumption
   about your folder structure.

Whichever one is used, it's printed to the terminal so it's never a
silent guess. To set your own default: copy `config.template.ini` to
`config.ini` (same folder) and fill in
`ground_truth_folder`/`prediction_folder` under `[paths]`. `config.ini`
is in `.gitignore`, so your local paths are never committed. Either entry
can be left blank; an invalid or nonexistent path is ignored (with a
warning) rather than breaking the tool, falling through to the next
option in the list above.

### Important: how pages are matched

The tool pairs up ground-truth and prediction files by **sorted
position**, not by filename - the first (alphabetically sorted) `.txt`
file in the GT folder is compared against the first sorted `.txt` file in
the prediction folder, and so on. Filenames themselves can differ between
the two folders.

This means you must make sure that:

- Both folders contain exactly one file per page, with no pages missing
  or duplicated.
- Files sort into the same page order in both folders. Zero-padded page
  numbers (`page_001.txt`, `page_002.txt`, ...) are a safe way to ensure
  this; `page_2.txt` sorting before `page_10.txt` is a common source of
  silent misalignment.

If a page is missing from one folder while an unrelated extra file exists
in the other, the file *counts* can still match while the actual page
pairing is wrong - the tool has no way to detect this, so double-check
your input folders before relying on the results.

### Non-interactive / scripted use

For running this as a step in another pipeline (rather than by hand),
there are two options that skip the folder-picker dialogs entirely:

**From the command line**, pass both folders as flags:

```bash
python main.py --gt /path/to/groundtruth --pred /path/to/predictions
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
[Output](#output)) is by default also printed to the terminal, which a
pipeline could easily miss. Every such message is therefore also
collected into `document_metrics["warnings"]`, so a caller can check for
problems without watching stdout. Pass `verbose=False` to suppress the
printing entirely (e.g. if your pipeline has its own logging) without
losing anything - the same information is still in
`document_metrics["warnings"]`.

## Output

Each run creates a new, dated subfolder next to the selected prediction
folder, named `evaluation_YYYY-MM-DD` (using the current date). Results
are never overwritten in place, so you can re-run the tool multiple times
per day/dataset without losing earlier results (running it twice on the
same day does overwrite that day's folder, though).

That folder contains:

- `evaluation_log.txt` - which GT/prediction folders were used, the
  evaluation date, and the list of files that were scored
- `metrics_pagewise.csv` / `metrics_pagewise.json` - CER/WER per page,
  both raw and jiwer-normalized
- `metrics_document.json` - a single nested summary: overall CER/WER plus
  the top recurring error characters (see below)
- `metrics_document_summary.csv` - the document-wide CER/WER summary as a
  flat table
- `metrics_document_top_raw.csv` / `metrics_document_top_normalized.csv` -
  the characters most often misrecognized, raw and normalized (only
  written if there are any attributable errors)
- `metrics_visualization.png` - a chart of CER/WER per page
- `evaluation_report.pdf` - a one-file report combining the summary, the
  top-error tables, and the chart, suitable for sharing or archiving

**Top error characters:** counts, per reference character, how often it
was substituted for the wrong character or dropped entirely (deletions),
then ranks the 10 most frequent offenders. Only alphabetic reference
characters are counted - digits, punctuation, and other symbols are
excluded. Extra characters the OCR *inserted* that don't correspond to
any ground truth character aren't attributed to anything.

**Empty-reference pages:** if a ground-truth page is empty but the OCR
still produced text (a hallucination), that page's CER/WER is reported as
explicitly **undefined/infinite** rather than a made-up percentage, and
shown in the chart/PDF as a gap in the line with a red "undefined"
marker. For exactly how this value is represented in each output file
(CSV, JSON, chart), see
[docs/METRICS.md](docs/METRICS.md#empty-reference-pages-exact-representation).

## Project layout

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
- `docs/METRICS.md` - detailed metrics reference: normalization pipelines,
  worked examples, and exact empty-reference representation
- `requirements.txt` - Python package requirements
- `README.md` - this file

## Requirements

- Python 3.10 or newer

## License

MIT - see [LICENSE](LICENSE).
