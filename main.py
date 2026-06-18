"""
author:    Ole Meiners
date:      2024-06-17
description: This script calculates the Character Error Rate (CER) and
 Word Error Rate (WER) between a set of reference texts (ground truth)
 and predicted texts (OCR outputs). It supports two evaluation regimes:
- raw whitespace tokenization without normalization
- normalized evaluation using jiwer default text transforms.
The results are saved in CSV and JSON formats, and a visualization of
 the per-page CER and WER is generated and saved as a PNG file.
"""

import glob
import os
from utils import select_folder, find_folder
from metrics import (
    calculate_lev_dist_text,
    calculate_lev_dist_words,
    calculate_jiwer_metrics,
    calculate_jiwer_document_level,
)
from plotting import save_and_plot_metrics

FOLDER_NAME_GT = "Goldstandard"
FOLDER_NAME_PRED = "Lumen-Lucernae"


def main():
    """Main function to execute the CER/WER calculation and create visualization."""
    # trying to find most likey folders for GT and predictions to use in folder selection dialog
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

    # get list with file pathes to zip together
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

    # counting variables for errors and text length
    lev_dist_raw_accumulated = 0
    lev_dist_raw_words_accumulated = 0
    chars_gt_accumulated = 0
    words_gt_accumulated = 0
    jiwer_references = []
    jiwer_predictions = []

    # list to collect per-page metrics
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

        # calculate cer and wer for case-sensitive eval schema
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
        jiwer_references.append(reference)
        jiwer_predictions.append(predicted)

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

    if jiwer_references and jiwer_predictions:
        (
            jiwer_cer_percentage,
            jiwer_wer_percentage,
        ) = calculate_jiwer_document_level(jiwer_references, jiwer_predictions)
    else:
        jiwer_cer_percentage = 0.0
        jiwer_wer_percentage = 0.0

    # Normalize the document-wide raw WER scores by dividing by the total reference word count.
    try:
        lev_dist_raw_words_accumulated /= words_gt_accumulated
    except ZeroDivisionError:
        print("Warning: No words in GT! Cannot calculate raw WER.")

    # calculate percentages
    cer_raw_percentage = lev_dist_raw_accumulated * 100
    wer_raw_percentage = lev_dist_raw_words_accumulated * 100

    print(
        "Results in total:\n"
        "Raw evaluation (whitespace tokenization):\n"
        f"\tCER raw: {cer_raw_percentage:.2f}%\n"
        f"\tWER raw: {wer_raw_percentage:.2f}%\n"
        "Normalized evaluation (jiwer default transforms):\n"
        f"\tCER normalized: {jiwer_cer_percentage:.2f}%\n"
        f"\tWER normalized: {jiwer_wer_percentage:.2f}%\n"
    )

    # Save metrics and create visualization (delegated to plotting helper)
    save_and_plot_metrics(page_metrics, folder_pred)


if __name__ == "__main__":
    main()
