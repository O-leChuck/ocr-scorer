"""Utility helpers for folder selection and metric export.

This module provides input/output helpers for the OCR evaluation
workflow, including selecting folders via dialog, locating relevant
folders on disk, and saving metric data to CSV/JSON.
"""

import json
import os
import glob
from datetime import date
from tkinter import Tk
from tkinter.filedialog import askdirectory
from tkinter.messagebox import showerror
import pandas as pd


def select_folder(initialdirectory, title):
    """Open a dialog to select a folder and return the selected path."""

    # we don't want a full GUI, so keep the root window from appearing
    root = Tk()
    root.withdraw()
    try:
        # show an "Open" dialog box and return the path to the selected folder
        folder_selected = askdirectory(initialdir=initialdirectory, title=title)
    finally:
        root.destroy()
    return folder_selected


def validate_and_select_folders(
    initial_dir_gt: str,
    initial_dir_pred: str,
    title_gt: str,
    title_pred: str,
) -> tuple[str, str] | None:
    """Repeatedly prompt user to select GT and prediction folders until
    file counts match. Allows user to abort at any time.

    Args:
        initial_dir_gt: Starting directory for GT folder selection
        initial_dir_pred: Starting directory for prediction folder selection
        title_gt: Title for GT folder selection dialog
        title_pred: Title for prediction folder selection dialog

    Returns:
        Tuple of (folder_gt, folder_pred) if valid, None if user aborts.
    """

    while True:
        # Select GT folder
        folder_gt = select_folder(initial_dir_gt, title_gt)
        if not folder_gt:  # User clicked Cancel
            print("Folder selection cancelled by user.")
            return None

        # Select prediction folder
        folder_pred = select_folder(initial_dir_pred, title_pred)
        if not folder_pred:  # User clicked Cancel
            print("Folder selection cancelled by user.")
            return None

        # Check if file counts match
        files_gt = sorted(glob.glob(os.path.join(folder_gt, "*.txt")))
        files_pred = sorted(glob.glob(os.path.join(folder_pred, "*.txt")))

        if len(files_gt) == len(files_pred):
            print(f"✓ Matched: {len(files_gt)} files in each folder")
            return (folder_gt, folder_pred)
        else:
            error_message = (
                f"File count mismatch!\n\n"
                f"Ground Truth folder: {len(files_gt)} files\n"
                f"Prediction folder: {len(files_pred)} files\n\n"
                f"Please ensure both folders contain the same dataset "
                f"and that the prediction process completed successfully.\n\n"
                f"Click OK to select the folders again, or Cancel to abort."
            )
            print(
                f"✗ Mismatch: GT has {len(files_gt)} files, "
                f"Prediction has {len(files_pred)} files."
            )
            # Show error dialog to user
            root = Tk()
            root.withdraw()
            showerror("Folder Validation Error", error_message)
            root.destroy()

            # Keep the last selected directories as defaults for next attempt
            initial_dir_gt = folder_gt
            initial_dir_pred = folder_pred


def find_folder(target_name, start_path="/home/"):
    """
    Search for a folder with the specified name starting from the given
    path.
    """

    for root, dirs, _ in os.walk(start_path):
        if target_name in dirs:
            return os.path.join(root, target_name)
    return None


def save_metrics(
    page_metrics: list[dict],
    output_dir: str,
    document_metrics: dict | None = None,
) -> pd.DataFrame:
    """Save per-page metrics to CSV/JSON and optionally document-level metrics.

    Outputs are written to the provided evaluation output directory.
    """
    df = pd.DataFrame(page_metrics)

    csv_output_path = os.path.join(output_dir, "metrics_pagewise.csv")
    df.to_csv(csv_output_path, index=False)
    print(f"\nMetrics saved to CSV: {csv_output_path}")

    json_output_path = os.path.join(output_dir, "metrics_pagewise.json")
    df.to_json(json_output_path, orient="records", indent=2)
    print(f"Metrics saved to JSON: {json_output_path}")

    if document_metrics is not None:
        save_document_metrics(document_metrics, output_dir)

    return df


def save_document_metrics(document_metrics: dict, output_dir: str) -> None:
    """Save document-level metrics and top-error metadata to CSV/JSON."""

    # Write a clean JSON summary (nested structure)
    json_output_path = os.path.join(output_dir, "metrics_document.json")
    with open(json_output_path, "w", encoding="utf-8") as json_file:
        json.dump(document_metrics, json_file, indent=2, ensure_ascii=False)
    print(f"Document metrics saved to JSON: {json_output_path}")

    # Export human-friendly CSV files: summary and two top-letter tables
    summary = document_metrics.get("summary", {})
    summary_rows = [{"metric": k, "value": v} for k, v in summary.items()]
    summary_csv_path = os.path.join(output_dir, "metrics_document_summary.csv")
    pd.DataFrame(summary_rows).to_csv(summary_csv_path, index=False)
    print(f"Document summary saved to CSV: {summary_csv_path}")

    def _write_top_table(section_name: str, filename: str):
        items = document_metrics.get(section_name, [])
        if not items:
            return
        table_rows = [
            {
                "rank": it.get("rank"),
                "character": it.get("character"),
                "count": it.get("count"),
                "percent": it.get("percent"),
            }
            for it in items
        ]
        out_path = os.path.join(output_dir, filename)
        pd.DataFrame(table_rows).to_csv(out_path, index=False)
        print(f"{section_name} saved to CSV: {out_path}")

    _write_top_table("top_error_chars_raw", "metrics_document_top_raw.csv")
    _write_top_table(
        "top_error_chars_normalized", "metrics_document_top_normalized.csv"
    )


def make_evaluation_output_folder(folder_pred: str) -> str:
    """
    Create a dated evaluation folder next to the prediction folder.
    """

    parent_dir = os.path.dirname(folder_pred)
    evaluation_folder = os.path.join(
        parent_dir, f"evaluation_{date.today().isoformat()}"
    )
    os.makedirs(evaluation_folder, exist_ok=True)
    return evaluation_folder


def save_evaluation_log(
    folder_gt: str,
    folder_pred: str,
    output_dir: str,
    evaluation_date: str | None = None,
) -> None:
    """
    Save an evaluation log containing folder metadata and evaluation
    date.
    """

    log_date = evaluation_date if evaluation_date else date.today().isoformat()
    gold_files = sorted(glob.glob(os.path.join(folder_gt, "*.txt")))
    pred_files = sorted(glob.glob(os.path.join(folder_pred, "*.txt")))
    txt_output_path = os.path.join(output_dir, "evaluation_log.txt")
    with open(txt_output_path, "w", encoding="utf-8") as txt_file:
        txt_file.write("Evaluation Log\n")
        txt_file.write("===============\n")
        txt_file.write(f"Evaluation date: {log_date}\n")
        txt_file.write(f"Goldstandard folder: {folder_gt}\n")
        txt_file.write(f"Prediction folder: {folder_pred}\n")
        txt_file.write(f"Export folder: {output_dir}\n")
        txt_file.write(f"Page count: {len(gold_files)}\n")
        txt_file.write("\nGround truth files:\n")
        for file_name in [os.path.basename(path) for path in gold_files]:
            txt_file.write(f"  - {file_name}\n")
        txt_file.write("\nPrediction files:\n")
        for file_name in [os.path.basename(path) for path in pred_files]:
            txt_file.write(f"  - {file_name}\n")
    print(f"Evaluation log saved to: {txt_output_path}")
