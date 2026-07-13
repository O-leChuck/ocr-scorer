"""Core CER/WER evaluation logic.

This module is the intended import point for using ocr_scorer as a
step in another pipeline:

    from ocr_scorer import run_evaluation
    output_dir, document_metrics = run_evaluation(gt_path, pred_path)

It deliberately does not import ocr_scorer.dialogs (tkinter) or
anything else GUI-related, so it - and anything that only needs
run_evaluation() - can be imported in a headless environment without
a display or even tkinter installed at all.
"""

import glob
import os
from datetime import date

from .io_helpers import (
    check_page_number_alignment,
    format_page_number_check_report,
    make_evaluation_output_folder,
    save_evaluation_log,
    save_metrics,
)
from .metrics import (
    aggregate_top_error_chars,
    calculate_lev_dist_text,
    calculate_lev_dist_words,
    calculate_jiwer_counts,
    calculate_jiwer_metrics,
)
from .plotting import create_pdf_report, plot_metrics


def run_evaluation(
    folder_gt: str, folder_pred: str, *, verbose: bool = True
) -> tuple[str, dict]:
    """Run the full CER/WER evaluation for the given GT/prediction
    folders and return (output_dir, document_metrics).

    This is the programmatic entry point for using ocr-scorer as a step
    in another pipeline - import it directly rather than going through
    ocr_scorer.cli's interactive folder-selection dialogs:

        from ocr_scorer import run_evaluation
        output_dir, document_metrics = run_evaluation(gt_path, pred_path)

    Unlike the interactive CLI (which prints a message and exits for a
    human at a terminal), this raises ValueError on any problem with
    the input, since there is no interactive retry available to a
    caller.

    Anything noteworthy that happens during a run but doesn't stop it
    (a page that failed to read, a page-number mismatch, a page whose
    CER/WER came out undefined - see docs/METRICS.md, "Empty-reference
    pages: exact representation") is only ever printed by default,
    which a calling pipeline could
    easily miss if it isn't watching stdout. Every such message is
    therefore also collected into document_metrics["warnings"], so it
    can't be missed by inspecting the return value alone. Set
    verbose=False to suppress the printing (e.g. when embedding this in
    a pipeline that has its own logging) without losing anything - the
    same information is still in document_metrics["warnings"].

    Raises:
        ValueError: if either folder doesn't exist, contains no .txt
            files, the two folders contain different numbers of .txt
            files, or every file failed to read.
    """
    warnings: list[str] = []

    def log(message: str) -> None:
        if verbose:
            print(message)

    if not os.path.isdir(folder_gt):
        raise ValueError(f"Ground truth folder does not exist: {folder_gt}")
    if not os.path.isdir(folder_pred):
        raise ValueError(f"Prediction folder does not exist: {folder_pred}")

    # collect sorted page file lists so names align when zipped
    files_gt = sorted(glob.glob(os.path.join(folder_gt, "*.txt")))
    files_pred = sorted(glob.glob(os.path.join(folder_pred, "*.txt")))

    if not files_gt or not files_pred:
        raise ValueError(
            "No .txt files found in the ground truth and/or prediction "
            "folder. Nothing to evaluate."
        )
    if len(files_gt) != len(files_pred):
        raise ValueError(
            f"File count mismatch: ground truth folder has "
            f"{len(files_gt)} .txt files, prediction folder has "
            f"{len(files_pred)}. Both folders must contain the same "
            "number of files (see README, 'how pages are matched')."
        )

    # Best-effort sanity check: files are paired purely by sort position
    # (prediction filenames aren't guaranteed to resemble GT filenames),
    # so warn if page numbers extracted from filenames disagree.
    page_number_check = check_page_number_alignment(files_gt, files_pred)
    _, page_number_messages = page_number_check
    for line in format_page_number_check_report(*page_number_check):
        log(line)
    warnings.extend(page_number_messages)

    # create a dated evaluation output folder next to the prediction folder
    output_dir = make_evaluation_output_folder(folder_pred)
    evaluation_date = date.today().isoformat()
    save_evaluation_log(
        folder_gt,
        folder_pred,
        output_dir,
        evaluation_date,
        page_number_check=page_number_check,
        verbose=verbose,
    )

    # counting variables for document-wide totals
    # Raw totals are normalized at the end by total reference length.
    # Jiwer totals are aggregated from per-page edit counts and ref lengths.
    lev_dist_raw_accumulated = 0
    lev_dist_raw_words_accumulated = 0
    chars_gt_accumulated = 0
    words_gt_accumulated = 0
    jiwer_char_edits_accumulated = 0
    jiwer_ref_chars_accumulated = 0
    jiwer_word_edits_accumulated = 0
    jiwer_ref_words_accumulated = 0

    # list to collect per-page metrics for CSV/JSON export and plotting
    page_metrics = []
    references: list[str] = []
    predictions: list[str] = []

    for file_gt, file_pred in zip(files_gt, files_pred):
        log(
            f"calculate cer/wer of files:\nfile_gt: \t{file_gt},\n"
            f"\tfile_pred: {file_pred}"
        )
        try:
            with open(
                file_gt, "r", encoding="utf-8", errors="replace"
            ) as ref_file:
                reference = ref_file.read()
        except FileNotFoundError:
            message = f"File not found: {file_gt}"
            log(message)
            warnings.append(message)
            continue
        except OSError as e:
            message = f"Error reading {file_gt}: {e}"
            log(message)
            warnings.append(message)
            continue

        try:
            with open(
                file_pred, "r", encoding="utf-8", errors="replace"
            ) as pred_file:
                predicted = pred_file.read()
        except FileNotFoundError:
            message = f"File not found: {file_pred}"
            log(message)
            warnings.append(message)
            continue
        except OSError as e:
            message = f"Error reading {file_pred}: {e}"
            log(message)
            warnings.append(message)
            continue

        # Compute this page's metrics into locals first, and only commit
        # them to the running totals/page_metrics below if every
        # calculation succeeds - otherwise a failure partway through
        # (e.g. a third-party library choking on some pathological page)
        # would leave the accumulators inconsistent with page_metrics
        # (counted into some totals but not others, or not in
        # page_metrics at all). One bad page is skipped with a warning
        # rather than losing every already-processed page in the batch.
        try:
            # calculate raw CER/WER on the unnormalized page text
            lev_dist_raw_page = calculate_lev_dist_text(reference, predicted)

            if len(reference) > 0:
                cer_raw_page = (lev_dist_raw_page / len(reference)) * 100
            elif lev_dist_raw_page > 0:
                # empty reference, but the OCR still produced text: a
                # hallucination, mathematically an unbounded (x/0, x>0)
                # rate
                cer_raw_page = float("inf")
            else:
                # both reference and prediction are empty: a true 0/0,
                # there is nothing to measure on this page
                cer_raw_page = float("nan")

            lev_dist_raw_words_page, words_gt_file = calculate_lev_dist_words(
                reference, predicted
            )

            if words_gt_file > 0:
                wer_raw_page = (lev_dist_raw_words_page / words_gt_file) * 100
            elif lev_dist_raw_words_page > 0:
                wer_raw_page = float("inf")
            else:
                wer_raw_page = float("nan")

            cer_jiwer_page, wer_jiwer_page = calculate_jiwer_metrics(
                reference, predicted
            )
            # Count normalized edits and normalized reference lengths
            # per page. These totals are summed across pages for the
            # final aggregate.
            (
                jiwer_char_edits_page,
                jiwer_ref_chars_page,
                jiwer_word_edits_page,
                jiwer_ref_words_page,
            ) = calculate_jiwer_counts(reference, predicted)
        except Exception as exc:  # noqa: BLE001 - deliberately broad
            message = (
                f"Skipping page - CER/WER calculation failed for "
                f"'{file_gt}' / '{file_pred}': {exc!r}"
            )
            log(message)
            warnings.append(message)
            continue

        references.append(reference)
        predictions.append(predicted)
        lev_dist_raw_accumulated += lev_dist_raw_page
        chars_gt_accumulated += len(reference)
        lev_dist_raw_words_accumulated += lev_dist_raw_words_page
        words_gt_accumulated += words_gt_file
        jiwer_char_edits_accumulated += jiwer_char_edits_page
        jiwer_ref_chars_accumulated += jiwer_ref_chars_page
        jiwer_word_edits_accumulated += jiwer_word_edits_page
        jiwer_ref_words_accumulated += jiwer_ref_words_page

        # extract page name from file path
        page_name = os.path.splitext(os.path.basename(file_pred))[0]

        # store per-page metrics
        page_metrics.append(
            {
                "page": page_name,
                "cer_raw": cer_raw_page,
                "wer_raw": wer_raw_page,
                "cer_jiwer_normalized": cer_jiwer_page,
                "wer_jiwer_normalized": wer_jiwer_page,
            }
        )

    if not page_metrics:
        raise ValueError(
            "No pages could be read (all files failed to open). "
            "Nothing to evaluate."
        )

    # Document-wide raw CER/WER: divide summed edits by summed reference
    # length. In the fully degenerate case where every single page has an
    # empty reference, fall back to the same infinity/NaN convention used
    # per-page (see docs/METRICS.md, "Empty-reference pages: exact
    # representation").
    if chars_gt_accumulated > 0:
        cer_raw_percentage = (
            lev_dist_raw_accumulated / chars_gt_accumulated
        ) * 100
    elif lev_dist_raw_accumulated > 0:
        cer_raw_percentage = float("inf")
        message = (
            "Warning: No characters in any GT file, but predictions "
            "contain text - raw CER is undefined (infinite)."
        )
        log(message)
        warnings.append(message)
    else:
        cer_raw_percentage = float("nan")
        message = (
            "Warning: No characters in any GT or prediction file - raw "
            "CER is not defined."
        )
        log(message)
        warnings.append(message)

    if words_gt_accumulated > 0:
        wer_raw_percentage = (
            lev_dist_raw_words_accumulated / words_gt_accumulated
        ) * 100
    elif lev_dist_raw_words_accumulated > 0:
        wer_raw_percentage = float("inf")
        message = (
            "Warning: No words in any GT file, but predictions contain "
            "words - raw WER is undefined (infinite)."
        )
        log(message)
        warnings.append(message)
    else:
        wer_raw_percentage = float("nan")
        message = (
            "Warning: No words in any GT or prediction file - raw WER is "
            "not defined."
        )
        log(message)
        warnings.append(message)

    # Same convention for the jiwer-normalized aggregate, using our own
    # summed edit/reference counts (calculate_jiwer_counts), not jiwer's
    # own per-page cer()/wer() calls - see README for that distinction.
    if jiwer_ref_chars_accumulated > 0:
        jiwer_cer_percentage = (
            jiwer_char_edits_accumulated / jiwer_ref_chars_accumulated
        ) * 100
    elif jiwer_char_edits_accumulated > 0:
        jiwer_cer_percentage = float("inf")
        message = (
            "Warning: No normalized reference characters available, but "
            "normalized predictions contain text - jiwer CER is "
            "undefined (infinite)."
        )
        log(message)
        warnings.append(message)
    else:
        jiwer_cer_percentage = float("nan")
        message = (
            "Warning: No normalized reference or prediction characters "
            "available - jiwer CER is not defined."
        )
        log(message)
        warnings.append(message)

    if jiwer_ref_words_accumulated > 0:
        jiwer_wer_percentage = (
            jiwer_word_edits_accumulated / jiwer_ref_words_accumulated
        ) * 100
    elif jiwer_word_edits_accumulated > 0:
        jiwer_wer_percentage = float("inf")
        message = (
            "Warning: No normalized reference words available, but "
            "normalized predictions contain words - jiwer WER is "
            "undefined (infinite)."
        )
        log(message)
        warnings.append(message)
    else:
        jiwer_wer_percentage = float("nan")
        message = (
            "Warning: No normalized reference or prediction words "
            "available - jiwer WER is not defined."
        )
        log(message)
        warnings.append(message)

    log(
        "Results in total:\n"
        "Raw evaluation (whitespace tokenization):\n"
        f"\tCER raw: {cer_raw_percentage:.2f}%\n"
        f"\tWER raw: {wer_raw_percentage:.2f}%\n"
        "Normalized evaluation (custom jiwer transforms):\n"
        f"\tCER normalized: {jiwer_cer_percentage:.2f}%\n"
        f"\tWER normalized: {jiwer_wer_percentage:.2f}%\n"
    )

    # top letter errors for the whole dataset
    top_error_chars_raw = [
        {"rank": i + 1, "character": char, "count": count, "percent": pct}
        for i, (char, count, pct) in enumerate(
            aggregate_top_error_chars(
                references, predictions, normalize=False, top_n=10
            )
        )
    ]
    top_error_chars_normalized = [
        {"rank": i + 1, "character": char, "count": count, "percent": pct}
        for i, (char, count, pct) in enumerate(
            aggregate_top_error_chars(
                references, predictions, normalize=True, top_n=10
            )
        )
    ]

    document_metrics = {
        "summary": {
            "evaluation_date": evaluation_date,
            "cer_raw": round(cer_raw_percentage, 4),
            "wer_raw": round(wer_raw_percentage, 4),
            "cer_normalized": round(jiwer_cer_percentage, 4),
            "wer_normalized": round(jiwer_wer_percentage, 4),
            "page_count": len(page_metrics),
        },
        "top_error_chars_raw": top_error_chars_raw,
        "top_error_chars_normalized": top_error_chars_normalized,
        "warnings": warnings,
    }

    log("\nTop 10 raw character errors:")
    for item in top_error_chars_raw:
        log(
            f"\t{item['rank']}. '{item['character']}' — {item['count']} "
            f"errors ({item['percent']:.2f}%)"
        )

    log("\nTop 10 normalized character errors:")
    for item in top_error_chars_normalized:
        log(
            f"\t{item['rank']}. '{item['character']}' — {item['count']} "
            f"errors ({item['percent']:.2f}%)"
        )

    # Save page-wise metrics, document-level metrics, create the plot,
    # and build the PDF.
    df = save_metrics(
        page_metrics,
        output_dir,
        document_metrics=document_metrics,
        verbose=verbose,
    )
    plot_metrics(df, output_dir, verbose=verbose)
    create_pdf_report(df, document_metrics, output_dir, verbose=verbose)

    return output_dir, document_metrics
