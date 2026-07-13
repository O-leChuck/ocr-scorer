"""
author:    Ole Meiners
date:      2024-06-17
description: This script calculates the Character Error Rate (CER) and
 Word Error Rate (WER) between a set of reference texts (ground truth)
 and predicted texts (OCR outputs). It supports two evaluation regimes:
- raw whitespace tokenization without normalization
- normalized evaluation using a custom jiwer transform pipeline
The results are saved in CSV and JSON formats, and a visualization of
 the per-page CER and WER is generated and saved as a PNG file.
"""

import glob
import os
from datetime import date
from config import load_default_paths
from io_helpers import (
    check_page_number_alignment,
    find_folder,
    format_page_number_check_report,
    make_evaluation_output_folder,
    save_evaluation_log,
    save_metrics,
    validate_and_select_folders,
)
from metrics import (
    aggregate_top_error_chars,
    calculate_lev_dist_text,
    calculate_lev_dist_words,
    calculate_jiwer_counts,
    calculate_jiwer_metrics,
)
from plotting import create_pdf_report, plot_metrics

FOLDER_NAME_GT = "Goldstandard"
FOLDER_NAME_PRED = "Lumen-Lucernae"


def _resolve_initial_directory(
    configured_path, auto_detected_path, fallback_path, label
):
    """Pick the starting directory for a folder-selection dialog and
    report which source was used, so a stale/unexpected default is
    visible upfront rather than silently steering folder selection.

    Priority: config.ini > auto-detected folder > built-in fallback.
    """
    if configured_path:
        print(f"Using configured default for {label}: {configured_path}")
        return configured_path
    if auto_detected_path:
        print(
            f"Auto-detected default for {label}: {auto_detected_path} "
            "(no config.ini entry set - see config.template.ini)"
        )
        return auto_detected_path
    print(f"Using built-in fallback default for {label}: {fallback_path}")
    return fallback_path


def main():
    """
    Main function to execute the CER/WER calculation and create
    visualization.
    """

    configured_gt, configured_pred = load_default_paths()

    # Try to locate likely GT and prediction folders automatically.
    # If not found, the user is prompted to select folders manually.
    target_folder_gt = find_folder(FOLDER_NAME_GT)
    target_folder_pred = find_folder(FOLDER_NAME_PRED)

    initial_directory_gt = _resolve_initial_directory(
        configured_gt,
        target_folder_gt,
        "/home/covid10/Nextcloud/Lumen-Lucernae/sources",
        "Goldstandard folder",
    )
    initial_directory_pred = _resolve_initial_directory(
        configured_pred,
        target_folder_pred,
        "/home/covid10/Nextcloud/Lumen-Lucernae/predictions/",
        "prediction folder",
    )

    title_gt_selection = (
        "Select ANY ONE .txt file - the whole Goldstandard folder "
        "will be used"
    )
    title_pred_selection = (
        "Select ANY ONE .txt file - the whole predictions folder "
        "will be used"
    )

    # Validate and select folders with retry logic if counts don't match
    result = validate_and_select_folders(
        initial_directory_gt,
        initial_directory_pred,
        title_gt_selection,
        title_pred_selection,
    )

    if result is None:
        print("Operation aborted by user.")
        return

    folder_gt, folder_pred = result

    # collect sorted page file lists so names align when zipped
    files_gt = sorted(glob.glob(os.path.join(folder_gt, "*.txt")))
    files_pred = sorted(glob.glob(os.path.join(folder_pred, "*.txt")))

    if not files_gt:
        print(
            "No .txt files found in the selected folders. "
            "Nothing to evaluate."
        )
        return

    # Best-effort sanity check: files are paired purely by sort position
    # (prediction filenames aren't guaranteed to resemble GT filenames),
    # so warn if page numbers extracted from filenames disagree.
    page_number_check = check_page_number_alignment(files_gt, files_pred)
    for line in format_page_number_check_report(*page_number_check):
        print(line)

    # create a dated evaluation output folder next to the prediction folder
    output_dir = make_evaluation_output_folder(folder_pred)
    evaluation_date = date.today().isoformat()
    save_evaluation_log(
        folder_gt,
        folder_pred,
        output_dir,
        evaluation_date,
        page_number_check=page_number_check,
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
        print(
            f"calculate cer/wer of files:\nfile_gt: \t{file_gt},\n"
            f"\tfile_pred: {file_pred}"
        )
        try:
            with open(
                file_gt, "r", encoding="utf-8", errors="replace"
            ) as ref_file:
                reference = ref_file.read()
        except FileNotFoundError:
            print(f"File not found: {file_gt}")
            continue
        except OSError as e:
            print(f"Error reading {file_gt}: {e}")
            continue

        try:
            with open(
                file_pred, "r", encoding="utf-8", errors="replace"
            ) as pred_file:
                predicted = pred_file.read()
        except FileNotFoundError:
            print(f"File not found: {file_pred}")
            continue

        except OSError as e:
            print(f"Error reading {file_pred}: {e}")
            continue

        references.append(reference)
        predictions.append(predicted)

        # calculate raw CER/WER on the unnormalized page text
        # Raw values remain page-specific for plotting and document totals.
        lev_dist_raw_page = calculate_lev_dist_text(reference, predicted)
        lev_dist_raw_accumulated += lev_dist_raw_page
        chars_gt_accumulated += len(reference)

        if len(reference) > 0:
            cer_raw_page = (lev_dist_raw_page / len(reference)) * 100
        elif lev_dist_raw_page > 0:
            # empty reference, but the OCR still produced text: a
            # hallucination, mathematically an unbounded (x/0, x>0) rate
            cer_raw_page = float("inf")
        else:
            # both reference and prediction are empty: a true 0/0, there
            # is nothing to measure on this page
            cer_raw_page = float("nan")

        lev_dist_raw_words_page, words_gt_file = calculate_lev_dist_words(
            reference, predicted
        )
        lev_dist_raw_words_accumulated += lev_dist_raw_words_page
        words_gt_accumulated += words_gt_file

        if words_gt_file > 0:
            wer_raw_page = (lev_dist_raw_words_page / words_gt_file) * 100
        elif lev_dist_raw_words_page > 0:
            wer_raw_page = float("inf")
        else:
            wer_raw_page = float("nan")

        cer_jiwer_page, wer_jiwer_page = calculate_jiwer_metrics(
            reference, predicted
        )
        # Count normalized edits and normalized reference lengths per page.
        # These totals are summed across pages for the final aggregate.
        (
            jiwer_char_edits_page,
            jiwer_ref_chars_page,
            jiwer_word_edits_page,
            jiwer_ref_words_page,
        ) = calculate_jiwer_counts(reference, predicted)
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
        print(
            "No pages could be read (all files failed to open). "
            "Nothing to evaluate."
        )
        return

    # Document-wide raw CER/WER: divide summed edits by summed reference
    # length. In the fully degenerate case where every single page has an
    # empty reference, fall back to the same infinity/NaN convention used
    # per-page (see README, "Empty-reference pages").
    if chars_gt_accumulated > 0:
        cer_raw_percentage = (
            lev_dist_raw_accumulated / chars_gt_accumulated
        ) * 100
    elif lev_dist_raw_accumulated > 0:
        cer_raw_percentage = float("inf")
        print(
            "Warning: No characters in any GT file, but predictions "
            "contain text - raw CER is undefined (infinite)."
        )
    else:
        cer_raw_percentage = float("nan")
        print(
            "Warning: No characters in any GT or prediction file - raw "
            "CER is not defined."
        )

    if words_gt_accumulated > 0:
        wer_raw_percentage = (
            lev_dist_raw_words_accumulated / words_gt_accumulated
        ) * 100
    elif lev_dist_raw_words_accumulated > 0:
        wer_raw_percentage = float("inf")
        print(
            "Warning: No words in any GT file, but predictions contain "
            "words - raw WER is undefined (infinite)."
        )
    else:
        wer_raw_percentage = float("nan")
        print(
            "Warning: No words in any GT or prediction file - raw WER is "
            "not defined."
        )

    # Same convention for the jiwer-normalized aggregate, using our own
    # summed edit/reference counts (calculate_jiwer_counts), not jiwer's
    # own per-page cer()/wer() calls - see README for that distinction.
    if jiwer_ref_chars_accumulated > 0:
        jiwer_cer_percentage = (
            jiwer_char_edits_accumulated / jiwer_ref_chars_accumulated
        ) * 100
    elif jiwer_char_edits_accumulated > 0:
        jiwer_cer_percentage = float("inf")
        print(
            "Warning: No normalized reference characters available, but "
            "normalized predictions contain text - jiwer CER is "
            "undefined (infinite)."
        )
    else:
        jiwer_cer_percentage = float("nan")
        print(
            "Warning: No normalized reference or prediction characters "
            "available - jiwer CER is not defined."
        )

    if jiwer_ref_words_accumulated > 0:
        jiwer_wer_percentage = (
            jiwer_word_edits_accumulated / jiwer_ref_words_accumulated
        ) * 100
    elif jiwer_word_edits_accumulated > 0:
        jiwer_wer_percentage = float("inf")
        print(
            "Warning: No normalized reference words available, but "
            "normalized predictions contain words - jiwer WER is "
            "undefined (infinite)."
        )
    else:
        jiwer_wer_percentage = float("nan")
        print(
            "Warning: No normalized reference or prediction words "
            "available - jiwer WER is not defined."
        )

    print(
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
    }

    print("\nTop 10 raw character errors:")
    for item in top_error_chars_raw:
        print(
            f"\t{item['rank']}. '{item['character']}' — {item['count']} "
            f"errors ({item['percent']:.2f}%)"
        )

    print("\nTop 10 normalized character errors:")
    for item in top_error_chars_normalized:
        print(
            f"\t{item['rank']}. '{item['character']}' — {item['count']} "
            f"errors ({item['percent']:.2f}%)"
        )

    # Save page-wise metrics, document-level metrics, create the plot,
    # and build the PDF.
    df = save_metrics(
        page_metrics, output_dir, document_metrics=document_metrics
    )
    plot_metrics(df, output_dir)
    create_pdf_report(df, document_metrics, output_dir)


if __name__ == "__main__":
    main()
