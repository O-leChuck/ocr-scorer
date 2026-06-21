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
from io_helpers import select_folder, find_folder, save_metrics
from metrics import (
    calculate_lev_dist_text,
    calculate_lev_dist_words,
    calculate_jiwer_counts,
    calculate_jiwer_metrics,
)
from plotting import plot_metrics

FOLDER_NAME_GT = "Goldstandard"
FOLDER_NAME_PRED = "Lumen-Lucernae"


def main():
    """Main function to execute the CER/WER calculation and create visualization."""
    # Try to locate likely GT and prediction folders automatically.
    # If not found, the user is prompted to select folders manually.
    target_folder_gt = find_folder(FOLDER_NAME_GT)
    target_folder_pred = find_folder(FOLDER_NAME_PRED)

    initial_directory_gt = (
        target_folder_gt
        if target_folder_gt
        else "/home/covid10/Nextcloud/Lumen-Lucernae/sources"
    )
    title_gt_selection = "Select a folder with Goldstandard Evaluation Data"
    folder_gt = select_folder(initial_directory_gt, title_gt_selection)

    initial_directory_pred = (
        target_folder_pred
        if target_folder_pred
        else "/home/covid10/Nextcloud/Lumen-Lucernae/predictions/"
    )
    title_pred_selection = "Select a folder with Predictions to evaluate"
    folder_pred = select_folder(initial_directory_pred, title_pred_selection)

    # collect sorted page file lists so names align when zipped
    # note: zip() below ignores any unmatched files if folder counts differ
    files_gt = glob.glob(os.path.join(folder_gt, "*.txt"))
    files_pred = glob.glob(os.path.join(folder_pred, "*.txt"))

    # check if number of files is the same, throw warning if not
    if len(files_gt) != len(files_pred):
        print(
            "Warning: Number of files in GT folder "
            f"({len(files_gt)}) and Prediction folder "
            f"({len(files_pred)}) do not match! Make sure to select the "
            "correct folders."
        )
        # TODO: implement a way in which hereafter user needs to select folder again

    # sort lists to make sure correct files are compared
    files_gt.sort()
    files_pred.sort()

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

    for file_gt, file_pred in zip(files_gt, files_pred):
        print(
            f"calculate cer/wer of files:\nfile_gt: \t{file_gt},\n\tfile_pred: {file_pred}"
        )
        try:
            with open(file_gt, "r", encoding="utf-8", errors="replace") as ref_file:
                reference = ref_file.read()
        except FileNotFoundError:
            print(f"File not found: {file_gt}")
            continue
        except Exception as e:
            print(f"Error reading {file_gt}: {e}")
            continue

        try:
            with open(file_pred, "r", encoding="utf-8", errors="replace") as pred_file:
                predicted = pred_file.read()
        except FileNotFoundError:
            print(f"File not found: {file_pred}")
            continue

        except Exception as e:
            print(f"Error reading {file_pred}: {e}")
            continue

        # calculate raw CER/WER on the unnormalized page text
        # Raw values remain page-specific for plotting and document totals.
        lev_dist_raw_page = calculate_lev_dist_text(reference, predicted)
        lev_dist_raw_accumulated += lev_dist_raw_page
        chars_gt_accumulated += len(reference)

        cer_raw_page = (
            lev_dist_raw_page / len(reference)
            if len(reference) > 0
            else lev_dist_raw_page
        ) * 100

        lev_dist_raw_words_page, words_gt_file = calculate_lev_dist_words(
            reference, predicted
        )
        lev_dist_raw_words_accumulated += lev_dist_raw_words_page
        words_gt_accumulated += words_gt_file

        wer_raw_page = (
            lev_dist_raw_words_page / words_gt_file
            if words_gt_file > 0
            else lev_dist_raw_words_page
        ) * 100

        cer_jiwer_page, wer_jiwer_page = calculate_jiwer_metrics(reference, predicted)
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

    # Normalize the document-wide raw CER scores by dividing by the total reference character count.
    try:
        lev_dist_raw_accumulated /= chars_gt_accumulated
    except ZeroDivisionError:
        print("Warning: No characters in GT! Cannot calculate raw CER.")

    # Normalize the document-wide raw WER scores by dividing by the total reference word count.
    try:
        lev_dist_raw_words_accumulated /= words_gt_accumulated
    except ZeroDivisionError:
        print("Warning: No words in GT! Cannot calculate raw WER.")

    # Aggregate jiwer-normalized counts across pages instead of concatenate text.
    try:
        jiwer_cer_percentage = (
            jiwer_char_edits_accumulated / jiwer_ref_chars_accumulated
        ) * 100
    except ZeroDivisionError:
        jiwer_cer_percentage = 0.0
        print("Warning: No normalized reference characters available for jiwer CER.")

    try:
        jiwer_wer_percentage = (
            jiwer_word_edits_accumulated / jiwer_ref_words_accumulated
        ) * 100
    except ZeroDivisionError:
        jiwer_wer_percentage = 0.0
        print("Warning: No normalized reference words available for jiwer WER.")

    # calculate percentages
    cer_raw_percentage = lev_dist_raw_accumulated * 100
    wer_raw_percentage = lev_dist_raw_words_accumulated * 100

    print(
        "Results in total:\n"
        "Raw evaluation (whitespace tokenization):\n"
        f"\tCER raw: {cer_raw_percentage:.2f}%\n"
        f"\tWER raw: {wer_raw_percentage:.2f}%\n"
        "Normalized evaluation (custom jiwer transforms):\n"
        f"\tCER normalized: {jiwer_cer_percentage:.2f}%\n"
        f"\tWER normalized: {jiwer_wer_percentage:.2f}%\n"
    )

    # Save page-wise metrics and plot the results from the same exported DataFrame.
    df = save_metrics(page_metrics, folder_pred)
    plot_metrics(df, folder_pred)


if __name__ == "__main__":
    main()
