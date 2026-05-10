"""
author:    Ole Meiners
date:      2024-06-17
description: This script calculates the Character Error Rate (CER) and Word Error Rate (WER) between a set of reference texts (ground truth) and predicted texts (OCR outputs). It supports both case-sensitive and case-insensitive evaluation schemas. The results are saved in CSV and JSON formats, and a visualization of the per-page CER and WER is generated and saved as a PNG file.
"""

import glob
import os
import Levenshtein
import pandas as pd
import matplotlib.pyplot as plt
from utils import select_folder, find_folder

FOLDER_NAME_GT = "Goldstandard"
FOLDER_NAME_PRED = "Lumen-Lucernae"


def calculate_lev_dist_text(reference: str, predicted: str) -> float:
    """Calculates the Character Error Rate (CER) between a reference text and a predicted text."""

    # is this actually necesary...? only if we split before...
    # predicted = " ".join(predicted)
    # reference = " ".join(reference)

    # Calculates CER using the Levenshtein distance
    lev_dist_text = Levenshtein.distance(predicted, reference)

    return lev_dist_text


def calculate_lev_dist_words(reference: str, predicted: str) -> tuple[float, int]:
    """Calculates the Word Error Rate (WER) between a reference text and a predicted text."""

    # Splits the predicted and reference texts into words
    predicted_words = predicted.split()
    reference_words = reference.split()

    # Calculates WER using the Levenshtein distance on words
    lev_dist_words = Levenshtein.distance(predicted_words, reference_words)

    return lev_dist_words, len(reference_words)


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
            f"Warning: Number of files in GT folder ({len(files_gt)}) and Prediction folder ({len(files_pred)}) do not match! Make sure to select the correct folders."
        )
        # TODO: implement a way in which hereafter user needs to select folder again

    # sort lists to make sure correct files are compared
    files_gt.sort()
    files_pred.sort()

    # counting varibles for errors and text length
    lev_dist_accumulated = 0
    lev_dist_words_accumulated = 0
    lev_dist_cs_accumulated = 0
    lev_dist_cs_words_accumulated = 0
    chars_gt_accumulated = 0
    words_gt_accumulated = 0

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
        lev_dist_cs_page = calculate_lev_dist_text(reference, predicted)
        lev_dist_cs_accumulated += lev_dist_cs_page
        chars_gt_accumulated += len(reference)

        cer_cs_page = (
            lev_dist_cs_page / len(reference)
            if len(reference) > 0
            else lev_dist_cs_page
        ) * 100

        lev_dist_cs_words_page, words_gt_file = calculate_lev_dist_words(
            reference, predicted
        )
        lev_dist_cs_words_accumulated += lev_dist_cs_words_page
        words_gt_accumulated += words_gt_file

        wer_cs_page = (
            lev_dist_cs_words_page / words_gt_file
            if words_gt_file > 0
            else lev_dist_cs_words_page
        ) * 100

        # calculate cer and wer for non-case-sensitive eval schema
        # lowercase reference and prediction
        reference_lower = reference.lower()
        predicted_lower = predicted.lower()

        lev_dist_page = calculate_lev_dist_text(reference_lower, predicted_lower)
        lev_dist_accumulated += lev_dist_page

        cer_page = (
            lev_dist_page / len(reference_lower)
            if len(reference_lower) > 0
            else lev_dist_page
        ) * 100

        lev_dist_words_page, words_gt_file = calculate_lev_dist_words(
            reference_lower, predicted_lower
        )
        lev_dist_words_accumulated += lev_dist_words_page

        wer_page = (
            lev_dist_words_page / words_gt_file
            if words_gt_file > 0
            else lev_dist_words_page
        ) * 100

        # extract page name from file path
        page_name = os.path.splitext(os.path.basename(file_pred))[0]

        # store per-page metrics
        page_metrics.append(
            {
                "page": page_name,
                "cer_case_sensitive": cer_cs_page,
                "wer_case_sensitive": wer_cs_page,
                "cer_case_insensitive": cer_page,
                "wer_case_insensitive": wer_page,
            }
        )

    # Normalize the document-wide CER scores by dividing it by the length of the reference text
    try:
        lev_dist_cs_accumulated /= chars_gt_accumulated
    except ZeroDivisionError:
        print("Warning: No characters in GT! Cannot calculate case-sensitive CER.")

    try:
        lev_dist_accumulated /= chars_gt_accumulated
    except ZeroDivisionError:
        print("Warning: No characters in GT! Cannot calculate CER.")

    # Normalize the document-wide WER scores by dividing it by the number of words in the GT
    try:
        lev_dist_cs_words_accumulated /= words_gt_accumulated
    except ZeroDivisionError:
        print("Warning: No words in GT! Cannot calculate case-sensitive WER.")

    try:
        lev_dist_words_accumulated /= words_gt_accumulated
    except ZeroDivisionError:
        print("Warning: No words in GT! Cannot calculate WER.")

    # calculate percentages
    cer_cs_percentage = lev_dist_cs_accumulated * 100
    cer_percentage = lev_dist_accumulated * 100
    wer_cs_percentage = lev_dist_cs_words_accumulated * 100
    wer_percentage = lev_dist_words_accumulated * 100

    print(
        "Results in total:\n"
        "Case-sensitive evaluation schema:\n"
        f"\tCER: {cer_cs_percentage:.2f}%\n"
        f"\tWER: {wer_cs_percentage:.2f}%\n"
        "Non-case-sensitive evaluation schema:\n"
        f"\tCER: {cer_percentage:.2f}%\n"
        f"\tWER: {wer_percentage:.2f}%\n"
    )

    # Create DataFrame from page metrics
    df = pd.DataFrame(page_metrics)

    # Save metrics to CSV
    csv_output_path = os.path.join(folder_pred, "../metrics_pagewise.csv")
    df.to_csv(csv_output_path, index=False)
    print(f"\nMetrics saved to CSV: {csv_output_path}")

    # Save metrics to JSON
    json_output_path = os.path.join(folder_pred, "../metrics_pagewise.json")
    df.to_json(json_output_path, orient="records", indent=2)
    print(f"Metrics saved to JSON: {json_output_path}")

    # Create visualization
    fig, ax = plt.subplots(figsize=(14, 7))

    # Plot all four metrics
    ax.plot(
        df.index,
        df["cer_case_sensitive"],
        marker="o",
        linewidth=2,
        label="CER (Case-Sensitive)",
        color="#FF6B6B",
    )
    ax.plot(
        df.index,
        df["wer_case_sensitive"],
        marker="s",
        linewidth=2,
        label="WER (Case-Sensitive)",
        color="#4ECDC4",
    )
    ax.plot(
        df.index,
        df["cer_case_insensitive"],
        marker="^",
        linewidth=2,
        label="CER (Case-Insensitive)",
        color="#FFE66D",
    )
    ax.plot(
        df.index,
        df["wer_case_insensitive"],
        marker="d",
        linewidth=2,
        label="WER (Case-Insensitive)",
        color="#95E1D3",
    )

    # Set labels and title
    ax.set_xlabel("Page Number", fontsize=12, fontweight="bold")
    ax.set_ylabel("Error Rate (%)", fontsize=12, fontweight="bold")
    ax.set_title(
        "CER/WER per Page - Case-Sensitive vs Case-Insensitive",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(len(df)))

    # Save figure
    chart_output_path = os.path.join(folder_pred, "../metrics_visualization.png")
    plt.tight_layout()
    plt.savefig(chart_output_path, dpi=300, bbox_inches="tight")
    print(f"Visualization saved to: {chart_output_path}")
    plt.show()


if __name__ == "__main__":
    main()
